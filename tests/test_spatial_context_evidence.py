"""Authored arrays/structural sizing only; no native inputs or production imports."""
from pathlib import Path
import base64
import copy
import hashlib
import itertools
import json
import math
import pickle
import struct

import numpy as np
import pytest

from scripts.evaluation import spatial_context_evidence as ev

scd=ev.scd
CENTERS=((40,40),(136,40),(40,136),(136,136))
SHAPE=(256,256)


def fixture(patterns=("H","V","D","E"),blank=False):
    frames=[]
    for dx,dy in ((0,0),(16,32)):
        pixels=np.zeros((*SHAPE,3),np.uint8);mask=np.zeros(SHAPE,bool)
        for (x,y),pattern in zip(CENTERS,patterns):
            x+=dx;y+=dy
            for row,col in itertools.product(range(3),repeat=2):
                for v,u in itertools.product(range(16),repeat=2):
                    name="H" if row*3+col==4 else pattern
                    value={"H":64+4*u,"V":64+4*v,"D":32+4*(u+v),"E":96+4*(u-v)}[name]
                    if blank and row*3+col==4:value=100
                    px=x-24+col*16+u;py=y-24+row*16+v
                    pixels[py,px]=value;mask[py,px]=True
        frames.append((pixels,mask,np.ones(SHAPE,bool)))
    return frames


def provenance(frames):
    runtime={"implementation":"authored-synthetic","numpy":"synthetic-pinned"}
    sources=[]
    for i,frame in enumerate(frames):
        p=frame.plan
        sources.append({"frame_id":f"synthetic-{i}","encoded_sha256":ev.sha(f"symbolic-encoded-{i}".encode()),"encoded_length":0,
            "decoded_shape":[*p.shape,3],"decoded_dtype":"|u1","decoded_rgb_sha256":p.frame_sha256,
            "ingress_sha256":ev.sha(f"symbolic-ingress-{i}".encode()),"endpoint":[f"synthetic-{i}",i],
            "region_id":ev.sha(f"symbolic-region-{i}".encode()),"packed_mask_sha256":ev.sha(f"symbolic-packed-{i}".encode()),
            "full_native_bool_sha256":p.source_mask_sha256,"usable_mask_sha256":p.usable_mask_sha256,
            "usable_helper_sha256":ev.sha(b"symbolic-usable-helper")})
    binding={"sources":sources,"source_pins":[["scripts/evaluation/spatial_context_descriptor.py",ev.ACCEPTED_SOURCE_SHA256]],
        "diagnostic_outer_sha256":ev.sha(b"symbolic-outer"),"diagnostic_nested_sha256":ev.sha(b"symbolic-nested"),
        "contract_sha256":ev.CONTRACT_SHA256,"source_sha256":ev.ACCEPTED_SOURCE_SHA256,
        "runtime_sha256":ev.sha(ev.canonical(runtime)),"authorization_sha256":ev.sha(b"symbolic-not-an-authorization")}
    binding["binding_sha256"]=ev.sha(ev.canonical(binding))
    return binding,runtime


def shared(arrays):
    return [ev.shared_frame(*a,role=i) for i,a in enumerate(arrays)]


def make_record(arrays=None):
    frames=shared(fixture() if arrays is None else arrays)
    binding,runtime=provenance(frames)
    return ev.paired(frames,binding,runtime)


@pytest.fixture(scope="module")
def positive():
    return make_record()


def reseal(record):
    record["record_sha256"]=ev.sha(ev.canonical({k:v for k,v in record.items() if k!="record_sha256"}))
    return ev.canonical(record)+b"\n"


def parity(arrays,record=None):
    frames=shared(arrays)
    binding,runtime=provenance(frames)
    record=ev.paired(frames,binding,runtime) if record is None else record
    for name,stored in zip(("A","B"),record["arms"]):
        old=[scd.describe_arm(a[0],f.plan.sites,name) for a,f in zip(arrays,frames)]
        new=[ev.derive(f,name) for f in frames]
        for o,n,f in zip(old,new,frames):
            assert o.tile_tables.tobytes()==n.tile_tables.tobytes()==f.tiles.tobytes()
            assert o.descriptors.tobytes()==n.descriptors.tobytes()
            assert o.sites==n.sites
            assert o.central_zero.tobytes()==n.central_zero.tobytes()
            assert o.neighbor_zero_counts.tobytes()==n.neighbor_zero_counts.tobytes()
            assert n.tile_tables is f.tiles and not f.tiles.flags.writeable
        match=scd.match_arms(*old,source_binding_sha256=scd.source_binding(*(f.plan for f in frames)))
        assert ev.compact_match(match,frames,new)==stored
        assert scd.support_report(*old)==scd.support_report(*new)
        assert ev.loads(ev.dumps(record))==record
    return frames,record


def test_positive_shared_identity_once_per_frame_and_exact_old_new_parity(monkeypatch):
    arrays=fixture();counts={"plans":0,"tiles":0,"luma":0};originals={n:getattr(scd,n) for n in ("plan_sites","tile_table","luma_from_rgb")}
    for name,key in (("plan_sites","plans"),("tile_table","tiles"),("luma_from_rgb","luma")):
        def traced(*args,_name=name,_key=key,**kwargs):
            counts[_key]+=1;return originals[_name](*args,**kwargs)
        monkeypatch.setattr(scd,name,traced)
    frames=shared(arrays);assert counts=={"plans":2,"tiles":8,"luma":2}
    identities=[id(f.tiles) for f in frames];hashes=[f.frozen_sha256 for f in frames]
    binding,runtime=provenance(frames);record=ev.paired(frames,binding,runtime)
    assert counts=={"plans":2,"tiles":8,"luma":2}
    assert identities==[id(f.tiles) for f in frames] and hashes==[ev.sha(f.tiles.tobytes()) for f in frames]
    assert record["outcomes"]==["no_admitted_pairs","candidate_discrimination_witness"]
    assert len(record["arms"][1]["pairs"])==4
    for name,original in originals.items():monkeypatch.setattr(scd,name,original)
    parity(arrays,record)


def authored_count_frame(count):
    """Contiguous authored footprints; no saved masks or image provenance."""
    shape=(900,1600);pixels=np.zeros((*shape,3),np.uint8)
    source=np.zeros(shape,bool);usable=np.ones(shape,bool)
    if count:
        columns=min(95,count);rows,extra=divmod(count,columns)
        height=48+16*(rows-1);width=48+16*(columns-1)
        source[:height,:width]=True
        if extra:source[height:height+16,:48+16*(extra-1)]=True
    return pixels,source,usable


def synthetic_collective(arrays,sealed_counts=(0,37,2570)):
    # Model the deferred driver's two gates, without implementing that driver.
    ev.require(sealed_counts==(0,37,2570),"synthetic_sealed_feasibility")
    receipts=[ev.prepare_shared_frame(*a,role=role) for a,role in zip(arrays,(0,0,1))]
    ev.require(tuple(len(r.plan.sites) for r in receipts)==sealed_counts,"synthetic_actual_feasibility")
    assert tuple(r.role for r in receipts)==(0,0,1)
    assert sum(r.plan.grid_centers for r in receipts)<=16800
    scd.check_site_caps(len(receipts[1].plan.sites),len(receipts[2].plan.sites))
    return receipts


def trace_prepared_work(monkeypatch):
    counts={"plans":0,"luma":0,"tiles":0,"contexts":0,"matches":0}
    for name,key in (("plan_sites","plans"),("luma_from_rgb","luma"),("tile_table","tiles"),
                     ("build_context_b","contexts"),("match_arms","matches")):
        original=getattr(scd,name)
        def traced(*args,_original=original,_key=key,**kwargs):
            counts[_key]+=1;return _original(*args,**kwargs)
        monkeypatch.setattr(scd,name,traced)
    return counts


def test_three_real_plans_then_only_two_matching_extractions(monkeypatch):
    arrays=[authored_count_frame(n) for n in (0,37,2570)]
    counts=trace_prepared_work(monkeypatch)
    receipts=synthetic_collective(arrays)
    assert counts=={"plans":3,"luma":0,"tiles":0,"contexts":0,"matches":0}
    assert [r.version for r in receipts]==[ev.PREPARED_FRAME_VERSION]*3
    calls=[];original=scd.tile_table
    def traced(luma,x,y):
        calls.append((counts["luma"],x,y));return original(luma,x,y)
    monkeypatch.setattr(scd,"tile_table",traced)
    frames=[ev.extract_prepared_frame(r) for r in receipts[1:]]
    assert counts=={"plans":3,"luma":2,"tiles":2607,"contexts":0,"matches":0}
    assert len(calls)==len(set(calls))==2607
    assert [len(f.plan.sites) for f in frames]==[37,2570]
    assert [f.plan for f in frames]==[r.plan for r in receipts[1:]]
    assert all(not a.flags.writeable for f in frames for a in (f.tiles,f.central_zero,f.neighbor_zero_counts))
    for arm in ("A","B"):
        derived=[ev.derive(f,arm) for f in frames]
        assert all(d.tile_tables is f.tiles for d,f in zip(derived,frames))
    assert counts["plans"]==3 and counts["luma"]==2 and counts["tiles"]==2607
    assert counts["contexts"]==2607 and counts["matches"]==0


@pytest.mark.parametrize("index",[0,1,2])
@pytest.mark.parametrize("gate",["sealed","actual"])
def test_collective_count_mismatch_refuses_before_dependent_work(monkeypatch,index,gate):
    counts=trace_prepared_work(monkeypatch);expected=(0,37,2570)
    altered=list(expected);altered[index]+=1
    arrays=[authored_count_frame(n) for n in (altered if gate=="actual" else expected)]
    with pytest.raises(ev.Refusal,match="synthetic_"+gate+"_feasibility"):
        synthetic_collective(arrays,tuple(altered) if gate=="sealed" else expected)
    assert counts=={"plans":0 if gate=="sealed" else 3,"luma":0,"tiles":0,"contexts":0,"matches":0}


@pytest.mark.parametrize("kind",["forged","copied","plan","role","version","pixels_ref","source_ref","usable_ref","plan_content","site_content"])
def test_prepared_receipt_tampering_refuses_before_extraction(monkeypatch,kind):
    arrays=fixture()[0];receipt=ev.prepare_shared_frame(*arrays)
    if kind=="forged":receipt=object.__new__(type(receipt))
    elif kind=="copied":
        clone=object.__new__(type(receipt))
        for field in ("_version","_role","_plan","_source_refs"):setattr(clone,field,getattr(receipt,field))
        receipt=clone
    elif kind=="plan":receipt._plan=copy.copy(receipt.plan)
    elif kind=="role":receipt._role=1
    elif kind=="version":receipt._version="spatial-context-prepared-frame-v0"
    elif kind.endswith("_ref"):
        index=("pixels_ref","source_ref","usable_ref").index(kind)
        refs=list(receipt._source_refs);refs[index]=refs[index].copy();receipt._source_refs=tuple(refs)
    elif kind=="plan_content":object.__setattr__(receipt.plan,"source_mask_sha256","f"*64)
    else:object.__setattr__(receipt.plan.sites[0],"site_id","f"*64)
    counts=trace_prepared_work(monkeypatch)
    with pytest.raises(ev.Refusal):ev.extract_prepared_frame(receipt)
    assert not any(counts.values())


@pytest.mark.parametrize("index",[0,1,2])
def test_each_prepared_input_mutation_refuses_and_consumes_receipt(monkeypatch,index):
    arrays=fixture()[0];receipt=ev.prepare_shared_frame(*arrays)
    saved=arrays[index].flat[0];arrays[index].flat[0]=1 if index==0 else not saved
    counts=trace_prepared_work(monkeypatch)
    with pytest.raises(ev.Refusal,match="input_mutation"):ev.extract_prepared_frame(receipt)
    arrays[index].flat[0]=saved
    with pytest.raises(ev.Refusal,match="prepared_receipt_unavailable"):ev.extract_prepared_frame(receipt)
    assert not any(counts.values())


def test_prepared_receipt_single_use_and_no_pickle_or_copy(monkeypatch):
    receipt=ev.prepare_shared_frame(*fixture()[0])
    for encode in (pickle.dumps,copy.copy,copy.deepcopy):
        with pytest.raises(ev.Refusal,match="prepared_receipt_serialization"):encode(receipt)
    counts=trace_prepared_work(monkeypatch);frame=ev.extract_prepared_frame(receipt)
    assert counts["plans"]==0 and counts["luma"]==1 and counts["tiles"]==len(frame.plan.sites)
    before=counts.copy()
    with pytest.raises(ev.Refusal,match="prepared_receipt_unavailable"):ev.extract_prepared_frame(receipt)
    assert counts==before
    with pytest.raises(ev.Refusal,match="prepared_receipt_authenticity"):ev.extract_prepared_frame(frame.plan)
    with pytest.raises(ev.Refusal,match="prepared_receipt_authenticity"):ev.extract_prepared_frame({"plan":frame.plan})
    assert counts==before


def test_prepared_receipt_process_binding_refuses_before_extraction(monkeypatch):
    receipt=ev.prepare_shared_frame(*fixture()[0]);process_id=ev.os.getpid()
    monkeypatch.setattr(ev.os,"getpid",lambda:process_id+1)
    counts=trace_prepared_work(monkeypatch)
    with pytest.raises(ev.Refusal,match="prepared_receipt_process"):ev.extract_prepared_frame(receipt)
    assert not any(counts.values())


@pytest.mark.parametrize("role,count",[(0,256),(0,257),(1,4096),(1,4097)])
def test_preparation_role_caps_with_real_planner_before_extraction(monkeypatch,role,count):
    arrays=authored_count_frame(count);counts=trace_prepared_work(monkeypatch)
    if count>(256,4096)[role]:
        with pytest.raises(ev.Refusal,match="shared_frame_site_limit"):ev.prepare_shared_frame(*arrays,role=role)
    else:
        receipt=ev.prepare_shared_frame(*arrays,role=role)
        assert len(receipt.plan.sites)==count and receipt.role==role
    assert counts=={"plans":1,"luma":0,"tiles":0,"contexts":0,"matches":0}


@pytest.mark.parametrize("kind",["role_bool","role_high","pixel_cap","pixels_dtype","source_dtype","usable_shape"])
def test_preparation_validation_refuses_before_expensive_work(monkeypatch,kind):
    arrays=list(fixture()[0]);role=0
    if kind=="role_bool":role=True
    elif kind=="role_high":role=2
    elif kind=="pixel_cap":arrays=[np.zeros((901,1,3),np.uint8),np.ones((901,1),bool),np.ones((901,1),bool)]
    elif kind=="pixels_dtype":arrays[0]=arrays[0].astype(np.float32)
    elif kind=="source_dtype":arrays[1]=arrays[1].astype(np.uint8)
    else:arrays[2]=arrays[2][:-1]
    counts=trace_prepared_work(monkeypatch)
    with pytest.raises(ev.Refusal):ev.prepare_shared_frame(*arrays,role=role)
    assert counts["plans"]==(1 if kind in ("source_dtype","usable_shape") else 0)
    assert counts["luma"]==counts["tiles"]==counts["contexts"]==counts["matches"]==0


def test_wrapper_prepared_and_old_arm_exact_bytes_flags_ids_and_matcher_parity():
    arrays=fixture();wrapped=shared(arrays)
    receipts=[ev.prepare_shared_frame(*a,role=i) for i,a in enumerate(arrays)]
    prepared=[ev.extract_prepared_frame(r) for r in receipts]
    for w,p in zip(wrapped,prepared):
        assert w.plan==p.plan and w.frozen_sha256==p.frozen_sha256
        for field in ("tiles","central_zero","neighbor_zero_counts"):
            assert getattr(w,field).tobytes()==getattr(p,field).tobytes()
            assert not getattr(p,field).flags.writeable
    binding,runtime=provenance(wrapped)
    assert ev.paired(wrapped,binding,runtime)==ev.paired(prepared,binding,runtime)
    for arm in ("A","B"):
        old=[scd.describe_arm(a[0],f.plan.sites,arm) for a,f in zip(arrays,wrapped)]
        derived=[ev.derive(f,arm) for f in prepared]
        for o,n in zip(old,derived):
            assert o.descriptors.tobytes()==n.descriptors.tobytes()
            assert o.tile_tables.tobytes()==n.tile_tables.tobytes()
            assert o.central_zero.tobytes()==n.central_zero.tobytes()
            assert o.neighbor_zero_counts.tobytes()==n.neighbor_zero_counts.tobytes()
            assert o.sites==n.sites
        source=scd.source_binding(*(f.plan for f in prepared))
        old_match=scd.match_arms(*old,source_binding_sha256=source)
        new_match=scd.match_arms(*derived,source_binding_sha256=source)
        assert old_match.distance_matrix.tobytes()==new_match.distance_matrix.tobytes()
        assert ev.compact_match(old_match,prepared,old)==ev.compact_match(new_match,prepared,derived)


@pytest.mark.parametrize("kind", ["periodic","blank","signed"])
def test_authored_negative_and_signed_edge_parity(kind):
    arrays=fixture(patterns=("H",)*4 if kind=="periodic" else ("H","V","D","E"),blank=kind=="blank")
    if kind=="signed":
        for pixels,mask,_ in arrays:pixels[mask]=255-pixels[mask]
        tile=np.repeat((255-17*np.arange(16,dtype=np.int64))[None,:,None],16,axis=0)
        tile=np.repeat(tile,3,axis=2).astype(np.uint8)
        gx,gy=scd.tile_gradients(scd.luma_from_rgb(tile))
        assert np.all(gx<0) and np.all(gy==0)
    _,record=parity(arrays)
    if kind=="periodic":assert record["outcomes"]==["no_admitted_pairs"]*2
    if kind=="blank":assert record["outcomes"]==["insufficient_support"]*2


@pytest.mark.parametrize("mask_kind", ["source","usable"])
def test_false_pixel_exclusion_and_outside_mask_invariance(mask_kind):
    arrays=fixture();arrays[0][1 if mask_kind=="source" else 2][40,40]=False
    frames,baseline=parity(arrays);assert len(frames[0].plan.sites)==3
    changed=[]
    for pixels,source,usable in arrays:
        other=pixels.copy();other[~source]=231;changed.append((other,source,usable))
    altered,new=parity(changed)
    for left,right in zip(frames,altered):
        assert left.tiles.tobytes()==right.tiles.tobytes()
        assert left.plan.frame_sha256!=right.plan.frame_sha256
    for old,new_arm in zip(baseline["arms"],new["arms"]):
        assert old["summary"]["forward_counts"]==new_arm["summary"]["forward_counts"]
        assert old["summary"]["distance_matrix_sha256"]==new_arm["summary"]["distance_matrix_sha256"]


def test_role_swap_and_aligned_site_permutation():
    arrays=fixture();frames,record=parity(arrays);_,reverse=parity(list(reversed(arrays)))
    old=record["arms"][1]["pairs"];new=reverse["arms"][1]["pairs"]
    assert {(p[1],p[2]) for p in old}=={(p[2],p[1]) for p in new}
    # Reconstruction returns the same canonical ordering regardless of supplied site enumeration.
    for frame,array in zip(frames,arrays):
        a=scd.describe_arm(array[0],frame.plan.sites,"B")
        b=scd.describe_arm(array[0],reversed(frame.plan.sites),"B")
        assert a.descriptors.tobytes()==b.descriptors.tobytes() and a.sites==b.sites


def domain_fixture(query,reference):
    arrays=fixture()
    for side,count in enumerate((query,reference)):
        pixels,source,usable=arrays[side]
        # Keep eligible sites while making excluded centers truly blank.
        source[:]=False
        for i,(x,y) in enumerate(CENTERS[:max(count,2)]):
            x+=(16 if side else 0);y+=(32 if side else 0)
            source[y-24:y+24,x-24:x+24]=True
            if i>=count:pixels[y-8:y+8,x-8:x+8]=100
    return arrays


@pytest.mark.parametrize("query,reference", [(0,0),(0,3),(3,0),(1,1),(1,2),(2,1)])
def test_v2_all_six_domain_cardinalities_exact_old_new_parity(query,reference):
    frames,record=parity(domain_fixture(query,reference))
    assert [int(np.count_nonzero(~f.central_zero)) for f in frames]==[query,reference]
    state="evaluated" if query and reference else "skipped_empty_domain"
    for arm in record["arms"]:
        assert arm["directed_state"]==state
        assert arm["summary"]["forward_counts"]["origins"]==query
        assert arm["summary"]["reverse_counts"]["origins"]==reference
        if state=="skipped_empty_domain":
            assert arm["forward"]==arm["reverse"]==arm["pairs"]==[]
            assert arm["summary"]["status"]=="insufficient_support"
            for direction in ("forward_counts","reverse_counts"):
                assert sum(v for k,v in arm["summary"][direction].items() if k!="origins")==0
            for w in (arm["summary"]["comparison_work"],arm["summary"]["witness"]["node_work"],arm["summary"]["witness"]["determinant_work"]):
                assert all(v==0 for k,v in w.items() if k!="limit")
        else:
            assert len(arm["forward"])==query and len(arm["reverse"])==reference
            assert arm["summary"]["comparison_work"]["processed"]==query*reference
            for rows,target in ((arm["forward"],reference),(arm["reverse"],query)):
                if target==1:assert all(row[1:]==[-1,-1,None,None,"absent_second_neighbor"] for row in rows)


@pytest.mark.parametrize("kind", ["state","rows","origins","status_counts","status","work","central"])
@pytest.mark.parametrize("empty", [True,False])
def test_resealed_v2_directed_state_and_independent_row_count_status_tampering(kind,empty):
    record=make_record(domain_fixture(0,3) if empty else fixture());arm=record["arms"][0]
    if kind=="state":arm["directed_state"]="evaluated" if empty else "skipped_empty_domain"
    elif kind=="rows":
        if empty:arm["reverse"]=[[0,-1,-1,None,None,"absent_second_neighbor"]]
        else:arm["forward"].pop()
    elif kind=="origins":arm["summary"]["forward_counts"]["origins"]+=1
    elif kind=="status_counts":arm["summary"]["forward_counts"]["ratio_accepted"]+=1
    elif kind=="status":arm["summary"]["status"]="no_admitted_pairs" if empty else "insufficient_support"
    elif kind=="work":arm["summary"]["comparison_work"]["requested"]+=1
    else:record["frames"][0]["sites"][0][3]=not record["frames"][0]["sites"][0][3]
    with pytest.raises(ev.Refusal):ev.loads(reseal(record))


@pytest.mark.parametrize("shape", [(0,9,32),(256,9,32),(4096,9,32)])
def test_table_exact_lossless_bits_signed_zeros_and_caps(shape):
    role=1 if shape[0]==4096 else 0
    table=np.ones(shape,dtype="<f4")
    if table.size:table.ravel()[0]=np.float32(-0.0);table.ravel()[-1]=np.nextafter(np.float32(0),np.float32(1))
    record=ev.encode_table(table,role);decoded=ev.decode_table(record,role)
    assert decoded.tobytes()==table.tobytes() and not decoded.flags.writeable
    assert record["raw_nbytes"]==1152*shape[0] and len(record["data"])==1536*shape[0]


@pytest.mark.parametrize("role,rows", [(0,257),(1,4097)])
def test_table_cap_plus_one_preallocation_and_prescan_guard(monkeypatch,role,rows):
    table=np.broadcast_to(np.zeros((1,9,32),dtype="<f4"),(rows,9,32))
    record={**ev.table_header(rows),"data":"","raw_sha256":"0"*64,"table_sha256":"0"*64}
    def unexpected(*args,**kwargs):raise AssertionError("finite scan or decode before capped header")
    monkeypatch.setattr(ev.np,"isfinite",unexpected);monkeypatch.setattr(ev.base64,"b64decode",unexpected)
    with pytest.raises(ev.Refusal):ev.encode_table(table,role)
    with pytest.raises(ev.Refusal):ev.decode_table(record,role)


@pytest.mark.parametrize("kind", ["dtype","order","shape","bool_shape","bytes","short","long","alphabet","padding","digest","header_digest","nan","infinity","endian"])
def test_table_mutations_refuse_atomically(kind):
    table=np.ones((1,9,32),dtype="<f4");record=ev.encode_table(table,0)
    if kind=="dtype":record["dtype"]="float32"
    elif kind=="order":record["order"]="F"
    elif kind=="shape":record["shape"]=[1,32,9]
    elif kind=="bool_shape":record["shape"][0]=True
    elif kind=="bytes":record["raw_nbytes"]-=1
    elif kind=="short":record["data"]=record["data"][:-1]
    elif kind=="long":record["data"]+="A"
    elif kind=="alphabet":record["data"]="!"+record["data"][1:]
    elif kind=="padding":record["data"]=record["data"][:-1]+"="
    elif kind=="digest":record["raw_sha256"]="f"*64
    elif kind=="header_digest":record["table_sha256"]="f"*64
    elif kind=="endian":record["dtype"]=">f4"
    else:
        table.ravel()[0]=np.nan if kind=="nan" else np.inf;raw=table.tobytes();header=ev.table_header(1)
        record.update(data=base64.b64encode(raw).decode(),raw_sha256=ev.sha(raw),table_sha256=ev.sha(ev.TABLE_DOMAIN+ev.canonical(header)+b"\x00"+raw))
    with pytest.raises(ev.Refusal):ev.decode_table(record,0)


@pytest.mark.parametrize("value", [0.0,-0.0,1.0,np.nextafter(0.0,1.0)])
def test_float64_distance_bit_roundtrip(value):
    encoded=ev.bits(value);assert len(encoded)==16 and encoded==encoded.lower()
    assert struct.pack("<d",ev.distance(encoded))==struct.pack("<d",value)


@pytest.mark.parametrize("value", [-1.0,float("inf"),float("nan")])
def test_invalid_distance_bits_refuse(value):
    with pytest.raises(ev.Refusal):ev.distance(struct.pack("<d",value).hex())


@pytest.mark.parametrize("kind", ["unknown","v1","site_id","site_index_bool","site_order","site_bin","domain","effective","runtime","source","binding","pair_id","pair_ref","pair_distance","support","witness","counter","table_zero"])
def test_resealed_internal_join_tampering_refuses(positive,kind):
    record=copy.deepcopy(positive);arm=record["arms"][1]
    if kind=="unknown":record["extra"]=None
    elif kind=="v1":record["schema"]="spatial-context-paired-evidence-v1"
    elif kind=="site_id":record["frames"][0]["sites"][0][0]="f"*64
    elif kind=="site_index_bool":arm["forward"][0][0]=False
    elif kind=="site_order":record["frames"][0]["sites"].reverse()
    elif kind=="site_bin":record["frames"][0]["sites"][0][5]=15
    elif kind=="domain":arm["summary"]["domain_hashes"][0]="f"*64
    elif kind=="effective":arm["summary"]["effective_hashes"][0]="f"*64
    elif kind=="runtime":record["runtime"]["implementation"]="changed"
    elif kind=="source":record["bindings"]["source_sha256"]="f"*64
    elif kind=="binding":record["bindings"]["sources"][0]["packed_mask_sha256"]="f"*64
    elif kind=="pair_id":arm["pairs"][0][0]="f"*64
    elif kind=="pair_ref":arm["pairs"][0][1]=4095
    elif kind=="pair_distance":arm["pairs"][0][3]=ev.bits(1.0)
    elif kind=="support":arm["summary"]["support"][0]["count"]+=1
    elif kind=="witness":arm["summary"]["witness"]["pairs"][0]=255
    elif kind=="counter":arm["summary"]["comparison_work"]["processed"]-=1
    else:record["frames"][0]["sites"][0][4]=8
    with pytest.raises(ev.Refusal):ev.loads(reseal(record))


@pytest.mark.parametrize("wire", [b'{"a":1,"a":2}\n',b'{"a":NaN}\n',b'{"a":Infinity}\n',b'{"a":1e100}\n',b'{"a":123456789012345678901}\n',b'{"a":"'+b'x'*1025+b'"}\n',b'['*17+b'0'+b']'*17+b'\n',b'[0,]\n'])
def test_lexical_duplicate_nonfinite_exponent_integer_string_depth_syntax_refusals(wire):
    with pytest.raises(ev.Refusal):ev.parse(wire)


def test_lexical_exact_depth_token_size_and_one_over_preparse_guards(monkeypatch):
    assert ev.parse(b'['*16+b'0'+b']'*16+b'\n')
    scanner=ev._Lexical(b'0\n');scanner.tokens=ev.TOKEN_CAP-1;scanner.scan();assert scanner.tokens==ev.TOKEN_CAP
    scanner=ev._Lexical(b'0\n');scanner.tokens=ev.TOKEN_CAP
    with pytest.raises(ev.Refusal,match="token_limit"):scanner.scan()
    # Global size guard is tested with the real cap before any JSON parser call.
    def unexpected(*args,**kwargs):raise AssertionError("JSON parse before input-size refusal")
    monkeypatch.setattr(ev.json,"loads",unexpected)
    with pytest.raises(ev.Refusal,match="evidence_byte_limit"):ev.parse(b' '* (ev.EVIDENCE_CAP+1))


def test_real_lexical_token_exact_cap_and_one_over():
    exact=b'[[]'+b',0'*131070+b']\n'
    scanner=ev._Lexical(exact);scanner.scan();assert scanner.tokens==262144
    over=b'[[],[]'+b',0'*131069+b']\n'
    with pytest.raises(ev.Refusal,match="token_limit"):ev._Lexical(over).scan()


def test_only_two_declared_base64_fields_allow_long_strings_and_escape_refusal(positive):
    assert ev.loads(ev.dumps(positive))==positive
    with pytest.raises(ev.Refusal,match="string_limit"):ev.parse(b'{"data":"'+b'A'*1025+b'"}\n')
    wire=ev.dumps(positive);start=wire.find(b'"data":"')+len(b'"data":"')
    altered=wire[:start]+b'\\u0041'+wire[start+1:]
    with pytest.raises(ev.Refusal,match="base64_wire_escape"):ev.parse(altered)
    with pytest.raises(ev.Refusal,match="wire_noncanonical"):ev.loads(wire+b'\n')


@pytest.mark.parametrize("kind,cap", [("site",128),("directed",256),("pair",192)])
def test_row_byte_exact_cap_and_cap_plus_one_before_large_serialization(kind,cap):
    # Capacity controls isolate byte bounds; semantic fixed-row schemas remain strict.
    exact=["x"*(cap-5)];assert len(ev.canonical(exact))+1==cap
    assert ev.row_size(exact,kind)==cap
    with pytest.raises(ev.Refusal,match="section_byte_limit|row_byte_limit"):ev.row_size(["x"*(cap-4)],kind)


@pytest.mark.parametrize("section,cap", [("source",262144),("summary",131072),("failure",65536),("framing",65536)])
def test_section_exact_cap_and_one_over_before_whole_object_dump(section,cap,monkeypatch):
    # Exact-sized lists of bounded ordinary strings, not alternate schema records.
    base=["x"*1000]*(cap//1003)
    remaining=cap-len(ev.canonical(base));base.append("x"*max(0,remaining-3))
    assert len(ev.canonical(base))==cap
    assert ev._bounded_object(base,cap)==cap
    base[-1]+="x"
    original=ev.canonical
    def scalar_only(value):
        assert not isinstance(value,(list,dict));return original(value)
    monkeypatch.setattr(ev,"canonical",scalar_only)
    with pytest.raises(ev.Refusal,match="section_byte_limit"):ev._bounded_object(base,cap)


@pytest.mark.parametrize("cap", [ev.EVIDENCE_CAP,ev.AGGREGATE_CAP,ev.MAX_MODEL,ev.MAX_PUBLICATION,ev.SMALL_CAP])
def test_file_aggregate_model_small_record_exact_cap_and_cap_plus_one(cap):
    ev.byte_guard(cap,cap,"test_cap")
    with pytest.raises(ev.Refusal,match="test_cap"):ev.byte_guard(cap+1,cap,"test_cap")


def test_immutable_input_and_shared_table_mutation_refusals(monkeypatch):
    arrays=fixture();original=scd.tile_table
    def mutated(*args,**kwargs):
        table=original(*args,**kwargs);arrays[0][0][0,0,0]=1;return table
    monkeypatch.setattr(scd,"tile_table",mutated)
    with pytest.raises(ev.Refusal,match="input_mutation"):ev.shared_frame(*arrays[0])
    monkeypatch.setattr(scd,"tile_table",original)
    frame=ev.shared_frame(*fixture()[0]);frame.tiles.setflags(write=True);frame.tiles.ravel()[0]+=1
    with pytest.raises(ev.Refusal,match="table_mutation"):ev.derive(frame,"A")


@pytest.mark.parametrize("source", [0,1,2])
def test_pixel_source_usable_mutation_after_shared_extraction_refuses(source):
    arrays=fixture()[0];frame=ev.shared_frame(*arrays)
    arrays[source].flat[0]=1 if source==0 else not arrays[source].flat[0]
    with pytest.raises(ev.Refusal,match="input_mutation"):ev.derive(frame,"A")


@pytest.mark.parametrize("fault", ["short_write","flush","fsync","readback","link"])
def test_failed_publication_preserves_no_false_completion_and_partial_state(tmp_path,positive,fault):
    path=tmp_path/'evidence.json';kwargs={}
    if fault=="short_write":kwargs["writer"]=lambda stream,wire:stream.write(wire[:-1])
    elif fault=="readback":kwargs["reader"]=lambda path,limit:b'corrupt\n'
    elif fault=="link":
        def fail(*args):raise OSError("inject link failure")
        kwargs["link"]=fail
    else:
        def fail(*args):raise OSError("inject persistence failure")
        kwargs[fault]=fail
    with pytest.raises(ev.Refusal):ev.publish(path,positive,**kwargs)
    assert not path.exists() and not (tmp_path/'completion.json').exists()


def test_publication_no_clobber_final_temporary_and_canonical_readback(tmp_path,positive):
    path=tmp_path/'evidence.json';out=ev.publish(path,positive)
    assert ev.read(path)==positive and out["bytes"]==path.stat().st_size
    assert out["aggregate_peak"]==2*path.stat().st_size
    original=path.read_bytes()
    with pytest.raises(ev.Refusal,match="publication_collision"):ev.publish(path,positive)
    assert path.read_bytes()==original
    other=tmp_path/'other.json';partial=tmp_path/'.other.json.partial';partial.write_bytes(b'prior')
    with pytest.raises(ev.Refusal,match="publication_collision"):ev.publish(other,positive)
    assert partial.read_bytes()==b'prior'


def test_aggregate_one_byte_over_refuses_before_write(tmp_path,positive,monkeypatch):
    wire=ev.dumps(positive);path=tmp_path/'evidence.json'
    monkeypatch.setattr(ev,"artifact_bytes",lambda _:ev.MAX_PUBLICATION-2*len(wire)+1)
    with pytest.raises(ev.Refusal,match="publication_model_limit"):ev.publish(path,positive)
    assert list(tmp_path.iterdir())==[]


def test_failed_final_readback_preserves_artifact_but_returns_no_completion(tmp_path,positive):
    def reader(path,limit):
        if not path.name.startswith('.'):return b'bad-final-readback'
        with path.open('rb') as stream:return stream.read(limit)
    with pytest.raises(ev.Refusal,match="publication_readback"):ev.publish(tmp_path/'evidence.json',positive,reader=reader)
    assert (tmp_path/'evidence.json').exists() and not (tmp_path/'completion.json').exists()


@pytest.mark.parametrize("stage", ["temporary","final"])
def test_oversized_publication_readback_capped_before_parse(tmp_path,positive,monkeypatch,stage):
    wire=ev.dumps(positive);target=tmp_path/'evidence.json';temporary=tmp_path/'.evidence.json.partial'
    reads=[];parsed=[];original_open=Path.open;original_loads=ev.loads
    class Traced:
        def __init__(self,stream):self.stream=stream
        def __enter__(self):self.stream.__enter__();return self
        def __exit__(self,*args):return self.stream.__exit__(*args)
        def read(self,limit):
            reads.append(limit);assert limit==len(wire)+1<=ev.EVIDENCE_CAP
            return self.stream.read(limit)
    def traced_open(path,mode='r',*args,**kwargs):
        stream=original_open(path,mode,*args,**kwargs)
        return Traced(stream) if mode=='rb' and path in (target,temporary) else stream
    def traced_loads(data):parsed.append(len(data));return original_loads(data)
    monkeypatch.setattr(Path,'open',traced_open);monkeypatch.setattr(ev,'loads',traced_loads)
    def enlarged_temporary(fd):
        ev.os.fsync(fd)
        with original_open(temporary,'ab') as stream:stream.write(b'x')
    def enlarged_final(source,destination):
        ev.os.link(source,destination)
        with original_open(destination,'ab') as stream:stream.write(b'x')
    kwargs={'fsync':enlarged_temporary} if stage=='temporary' else {'link':enlarged_final}
    with pytest.raises(ev.Refusal,match='publication_readback_length'):ev.publish(target,positive,**kwargs)
    assert reads==[len(wire)+1]*(1 if stage=='temporary' else 2)
    assert parsed==([] if stage=='temporary' else [len(wire)])
    assert not (tmp_path/'completion.json').exists()


@pytest.mark.parametrize("kind", ["determinants_low","nodes_above_combinations","found_rank","reversed_refs"])
def test_resealed_witness_accounting_traversal_and_order_tampering_refuses(positive,kind):
    record=copy.deepcopy(positive);w=record['arms'][1]['summary']['witness'];nw=w['node_work'];dw=w['determinant_work']
    if kind=='determinants_low':
        for key in ('requested','charged','processed'):dw[key]-=8
    elif kind=='nodes_above_combinations':
        for key in ('requested','charged','processed'):nw[key]+=1
        for key in ('requested','charged','processed'):dw[key]+=8
    elif kind=='found_rank':
        # Full structural fixture has many combinations; charge one extra node
        # coherently, retaining its original quartet and thereby wrong rank.
        record=structural_maximum();w=record['arms'][1]['summary']['witness'];nw=w['node_work'];dw=w['determinant_work']
        for key in ('requested','charged','processed'):nw[key]+=1
        for key in ('requested','charged','processed'):dw[key]+=8
    else:w['pairs'].reverse()
    with pytest.raises(ev.Refusal):ev.loads(reseal(record))


def test_quartet_rank_matches_streaming_small_combinations():
    for count in range(4,10):
        for rank,quad in enumerate(itertools.combinations(range(count),4)):
            assert ev.quartet_rank(quad,count)==rank


def structural_maximum():
    """Full shape only, with fabricated coherent nearest summaries, no appearance claim."""
    frames=[]
    for role,count in enumerate((256,4096)):
        shape=(900,1600);frame_sha=ev.sha(f"structural-frame-{role}".encode());mask_sha=ev.sha(f"structural-mask-{role}".encode());usable_sha=ev.sha(f"structural-usable-{role}".encode())
        points=list(itertools.islice(((x,y) for y in range(24,877,16) for x in range(24,1577,16)),count))
        sites=tuple(scd.Site(scd._site_id(frame_sha,mask_sha,usable_sha,x,y),x,y) for x,y in points)
        plan=scd.DomainPlan(scd.CONTRACT_VERSION,shape,frame_sha,mask_sha,usable_sha,0,count,sites)
        table=np.ones((count,9,32),dtype="<f4");table.setflags(write=False)
        zero=np.zeros(count,bool);neighbors=np.zeros(count,np.int64)
        frames.append(ev.SharedFrame(plan,table,zero,neighbors,ev.sha(table.tobytes())))
    binding,runtime=provenance(frames);runtime["implementation"]="structural-size-only";binding["runtime_sha256"]=ev.sha(ev.canonical(runtime));binding["binding_sha256"]=ev.sha(ev.canonical({k:v for k,v in binding.items() if k!="binding_sha256"}))
    arms=[];source=scd.source_binding(*(f.plan for f in frames));zero=ev.bits(0.0);one=ev.bits(1.0)
    for name in ("A","B"):
        derived=[ev.derive(f,name) for f in frames]
        forward=[[i,i,(i+1)%4096,zero,one,"ratio_accepted"] for i in range(256)]
        reverse=[[j,j%256,(j+1)%256,zero,one,"ratio_accepted"] for j in range(4096)]
        objects=sorted([scd.make_pair_record(frames[0].plan.sites[i].site_id,frames[1].plan.sites[i].site_id,frames[0].plan.sites[i].xy,frames[1].plan.sites[i].xy,0.0) for i in range(256)],key=lambda p:p["pair_id"])
        indices={p["pair_id"]:i for i,p in enumerate(objects)};nodes=scd.Budget(4096);determinants=scd.Budget(32768)
        witness=scd.common_four(objects,nodes,determinants,source);assert witness["status"]=="found"
        left_lookup={s.site_id:i for i,s in enumerate(frames[0].plan.sites)};right_lookup={s.site_id:i for i,s in enumerate(frames[1].plan.sites)}
        pairs=[[p["pair_id"],left_lookup[p["left_id"]],right_lookup[p["right_id"]],zero] for p in objects]
        def counts(n):return {"origins":n,"absent_second_neighbor":0,"nearest_distance_tie":0,"ratio_rejected":0,"ratio_accepted":n}
        budget=scd.Budget(1048576);budget.reserve(1048576);budget.processed=1048576
        arms.append({"arm":name,"directed_state":"evaluated","forward":forward,"reverse":reverse,"pairs":pairs,
            "summary":{"status":"candidate_discrimination_witness","forward_counts":counts(256),"reverse_counts":counts(4096),
            "witness":{"status":"found","pairs":[indices[i] for i in witness["pair_ids"]],"source_binding_sha256":source,"node_work":nodes.close(),"determinant_work":determinants.close()},
            "comparison_work":budget.close(),"support":[ev._compact_support(a) for a in derived],
            "domain_hashes":[ev.sha(ev.canonical([s.site_id for s in a.sites])) for a in derived],
            "effective_hashes":[ev.effective_hash(f,name,a.descriptors) for f,a in zip(frames,derived)],
            "distance_matrix_sha256":ev.sha(b"structural-size-only-not-a-distance-matrix")}})
    return ev.seal({"schema":ev.SCHEMA,"treatment":ev.TREATMENT,"bindings":binding,"runtime":runtime,"size_bound":ev.size_model(),
        "frames":[ev.frame_record(f,i) for i,f in enumerate(frames)],"arms":arms,"outcomes":["candidate_discrimination_witness"]*2,
        "counters":{"storage_replay":"no_matching_or_common_search"},"failure":{"code":"none"}})


def test_actual_maximum_structural_serialization_independent_bound_and_aggregate_peak(tmp_path,monkeypatch):
    def no_distances(*args,**kwargs):raise AssertionError("structural sizing silently invoked matching")
    monkeypatch.setattr(scd,"pairwise_squared",no_distances)
    record=structural_maximum();wire=ev.dumps(record)
    independent=1536*4352+128*4352+2*256*4352+2*192*256+524288
    assert independent==ev.MAX_MODEL==10092544
    assert 2*independent+3*65536==ev.MAX_PUBLICATION==20381696
    assert len(wire)<=ev.account(record)["accounted_upper_bound"]<=independent<ev.EVIDENCE_CAP
    assert ev.size_model(37,2570)["evidence"]==6211328
    for name in ('worker.log','completion.json','failure.json'):(tmp_path/name).write_bytes(b'x'*65536)
    measured=ev.publish(tmp_path/'evidence.json',record)
    assert measured["aggregate_peak"]==2*len(wire)+3*65536<=20381696<ev.AGGREGATE_CAP
    assert ev.read(tmp_path/'evidence.json')==record
    root=Path(__file__).resolve().parents[1]/'agent swarm instantiation template/runs'
    prefix='SOL-SPATIAL-CONTEXT-PREPARED-FRAME-01'
    with (root/(prefix+'.maximum-evidence.json')).open('xb') as stream:stream.write(wire)
    measurement={"structural_only":True,"appearance_or_physical_claim":False,"rows":[256,4096],"directed_rows":8704,"pair_rows":512,
        "evidence_bytes":len(wire),"evidence_sha256":ev.sha(wire),"aggregate_peak":measured["aggregate_peak"],
        "model_bound":independent,"publication_bound":20381696,"hard_caps":[16777216,37748736],"sections":ev.account(record),
        "actual_matching_comparisons_executed":0,"structural_comparison_counter_fields":"fabricated shape-consistent metadata; not execution telemetry"}
    with (root/(prefix+'.size-control.json')).open('xb') as stream:stream.write((json.dumps(measurement,sort_keys=True,indent=2)+'\n').encode('ascii'))


def test_module_boundary_no_native_driver_or_production_imports():
    text=Path(ev.__file__).read_text()
    assert 'mc_photogrammetry' not in text and 'cv2' not in text and 'SIFT' not in text
    assert 'def main(' not in text and 'subprocess' not in text
