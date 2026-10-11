"""Only stdlib fake providers; no cv2/NumPy/production imports."""
import ast
import builtins
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

PATH=Path(__file__).parents[1]/"scripts/evaluation/validate_cpu_support_reference.py"
spec=importlib.util.spec_from_file_location("_cpu_harness_test",PATH)
h=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=h
old_import=builtins.__import__
def forbidden(name,*args,**kwargs):
    if name.split(".")[0] in ("cv2","numpy","mc_photogrammetry"):
        raise AssertionError("native/production import prohibited")
    return old_import(name,*args,**kwargs)
builtins.__import__=forbidden
try:
    spec.loader.exec_module(h)
    m=h.symbolic()
finally:
    builtins.__import__=old_import


@pytest.fixture(autouse=True)
def fake_lifetime(monkeypatch):
    monkeypatch.setattr(m,"_LEDGER",m._AggregateLedger())  # test-only fresh worker model
    monkeypatch.setattr(builtins,"__import__",forbidden)



def test_validate(r,receipt,*,require_child=False):
    return h.validate_receipt(r,receipt,expected_attempt=receipt["record"]["attempt"],
                              expected_pid=os.getpid(),require_child=require_child)
test_validate.__test__=False


def fake_summary(r):
    return {"status":"SYNTHETIC_ONLY","request_sha":h.request_hash(r),"output_sha":"1"*64,
            "support_seal":"2"*64,"receipt_sha":"3"*64,"attempt":h.attempt_nonce(),"pid":os.getpid()}


def request():
    return h.make_request(next(h.matrix_recipes()))


def fail(code,call):
    with pytest.raises(h.Closed) as error:
        call()
    assert error.value.refusal.code==code


def test_pure_scope_and_real_path_closed():
    assert "cv2" not in sys.modules and "numpy" not in sys.modules
    assert not h.NATIVE_ATTESTOR_PINS
    fail("unsupported_attestor",h.native_preflight)
    tree=ast.parse(PATH.read_text())
    assert not any(isinstance(n,ast.Assert) for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Import) and any(a.name.startswith(("cv2","numpy","mc_photogrammetry")) for a in n.names) for n in ast.walk(tree))


@pytest.mark.parametrize("recipe",tuple(h.matrix_recipes()),ids=lambda x:x.name)
def test_finite_matrix_fake_call_and_sealed_support(recipe):
    r=h.make_request(recipe)
    provider=h.FakeProvider()
    receipt=h.Worker(provider).run(r)
    summary=test_validate(r,receipt,require_child=False)
    assert provider.calls==1
    assert summary["status"]=="SYNTHETIC_ONLY"
    assert receipt["record"]["kind"]==recipe.kind
    assert receipt["record"]["channels"]==recipe.channels
    assert m._LEDGER.used==receipt["record"]["worker_reservation"]
    assert h.expected_events(r)["scalar_pixels"]==r.destination.shape[0]*(recipe.width%h.expected_events(r)["block_pixels"])


def test_plan_reproducible_unique_bounded_and_all_contracts():
    recipes=tuple(h.matrix_recipes())
    assert len(recipes)==492
    assert len(set(r.name for r in recipes))==len(recipes)
    assert set((r.kind,r.channels) for r in recipes)=={("u8",1),("u8",3),("f32le",1)}
    assert h.matrix_pin()==h.matrix_pin()
    for dispatch,block in (("BASELINE",8),("SSE4_1",8),("AVX2",16)):
        assert {r.width for r in recipes if r.dispatch==dispatch}>={block-1,block,block+1}


@pytest.mark.parametrize("change,code",[
    ({"mode":"NATIVE"},"unsupported_attestor"),({"atomic":False},"missing_attestation"),
    ({"fp_policy":"unknown"},"missing_attestation"),({"entry":"another"},"branch_mismatch"),
    ({"dispatcher":"AVX512_SKX"},"branch_mismatch"),({"identity":{}},"identity_mismatch"),
])
def test_bad_capability_zero_backend_calls(change,code):
    class Bad(h.FakeProvider):
        def capability(self,r):
            result=super().capability(r); result.update(change); return result
    provider=Bad()
    fail(code,lambda:h.Worker(provider).run(request()))
    assert provider.calls==0


def test_label_only_is_not_capability_and_old_receipt_not_preflight():
    r=request(); provider=h.FakeProvider()
    fail("missing_attestation",lambda:h.check_capability(r,{"dispatcher":"BASELINE"}))
    fail("missing_attestation",lambda:h.check_capability(r,provider.evidence(r,"after")))


@pytest.mark.parametrize("change,code",[
    ({"phase":"after"},"event_order"),({"events":{}},"branch_mismatch"),
    ({"fp":{}},"fp_mismatch"),({"request_sha":"0"*64},"metadata_mismatch"),
    ({"identity":{}},"identity_mismatch"),({"mode":"NATIVE"},"unsupported_attestor"),
])
def test_actual_call_evidence_generated_at_call_and_vetoes_promotion(change,code):
    class Bad(h.FakeProvider):
        def evidence(self,r,phase):
            result=super().evidence(r,phase)
            if phase=="before": result.update(change)
            return result
    provider=Bad()
    fail(code,lambda:h.Worker(provider).run(request()))
    assert provider.calls==1  # capability passed; fresh actual evidence vetoed


@pytest.mark.parametrize("mxcsr",[0x9F80,0x1FC0,0x3F80,True,-1,0x10000])
def test_wrong_fp_controls_after_call(mxcsr):
    class Bad(h.FakeProvider):
        def evidence(self,r,phase):
            result=super().evidence(r,phase); result["fp"]["mxcsr"]=mxcsr; return result
    provider=Bad()
    fail("fp_mismatch",lambda:h.Worker(provider).run(request()))
    assert provider.calls==1


@pytest.mark.parametrize("field,change,code",[
    ("map_x",{"address":0x10000000},"unsupported_alias"),
    ("map_x",{"address":0x20000001},"unsupported_layout"),
    ("source",{"channels":2},"unsupported_layout"),
    ("destination",{"address":m.UINTPTR_MAX},"unsafe_address"),
])
def test_bad_layout_before_call(field,change,code):
    r=request(); r=replace(r,**{field:replace(getattr(r,field),**change)})
    provider=h.FakeProvider()
    fail(code,lambda:h.Worker(provider).run(r)); assert provider.calls==0


@pytest.mark.parametrize("bits",[1,0x7FC00000,0x7F800000,0xFF800000,0x4F000000])
def test_bad_coordinates_before_call(bits):
    r=request(); data=m.U32.pack(bits)+r.map_x.data[4:]
    r=replace(r,map_x=replace(r.map_x,data=data))
    provider=h.FakeProvider()
    fail("input_refusal",lambda:h.Worker(provider).run(r)); assert provider.calls==0


def test_one_worker_one_witness_even_on_failure_or_interruption():
    r=request(); provider=h.FakeProvider(); worker=h.Worker(provider)
    worker.run(r)
    fail("one_shot",lambda:worker.run(r)); assert provider.calls==1


@pytest.mark.parametrize("error",[RuntimeError,KeyboardInterrupt,SystemExit])
def test_worker_failure_and_interruption_release_ownership_but_not_reuse(error):
    class Bad(h.FakeProvider):
        def call(self,r):
            self.calls+=1; raise error()
    provider=Bad(); worker=h.Worker(provider)
    with pytest.raises(error): worker.run(request())
    fail("one_shot",lambda:worker.run(request()))
    assert not worker.lock.locked() and provider.calls==1 and m._LEDGER.used>0


def test_nested_worker_refuses_before_more_calls():
    class Nested(h.FakeProvider):
        def call(self,r):
            fail("ownership_busy",lambda:worker.run(r)); return super().call(r)
    provider=Nested(); worker=h.Worker(provider)
    worker.run(request()); assert provider.calls==1


@pytest.mark.parametrize("field,value",[("output_hex","ff"),("kind","f32le"),("channels",3),("stride",1),
                                          ("request_sha","0"*64),("status","PASS")])
def test_resealed_receipt_tamper(field,value):
    r=request(); receipt=h.Worker(h.FakeProvider()).run(r)
    receipt["record"][field]=value; receipt["seal"]=h.seal(receipt["record"])
    with pytest.raises(h.Closed): test_validate(r,receipt,require_child=False)


def test_digest_tamper_and_wrong_order():
    r=request(); receipt=h.Worker(h.FakeProvider()).run(r)
    receipt["seal"]="0"*64
    fail("receipt_mismatch",lambda:test_validate(r,receipt,require_child=False))
    receipt["record"]["before"],receipt["record"]["after"]=receipt["record"]["after"],receipt["record"]["before"]
    receipt["seal"]=h.seal(receipt["record"])
    fail("event_order",lambda:test_validate(r,receipt,require_child=False))


def test_exact_request_output_roundtrip_and_framing():
    r=request(); encoded=h.encode_request(r); decoded=h.decode_request(encoded)
    assert h.request_hash(r)==h.request_hash(decoded)
    assert decoded.source.data==r.source.data
    assert h.unframe(h.framed(encoded))==encoded
    for bad in (b"",b"123",h.framed(encoded)[:-1],h.framed(encoded)+b"x"):
        with pytest.raises(h.Closed): h.unframe(bad)


def test_parent_and_wire_cap_boundaries(monkeypatch):
    r=request(); planned=h.parent_reservation(r)[2]
    monkeypatch.setattr(h,"MAX_PARENT_BYTES",planned)
    assert h.parent_reservation(r)[2]==planned
    monkeypatch.setattr(h,"MAX_PARENT_BYTES",planned-1)
    fail("parent_cap",lambda:h.encode_request(r))
    monkeypatch.setattr(h,"MAX_WIRE_BYTES",3)
    assert h.unframe(h.framed(b"123"))==b"123"
    fail("wire_cap",lambda:h.framed(b"1234"))


def test_durable_marker_before_launch_failure_and_no_retry(tmp_path):
    path=tmp_path/"marker"; guard=h.OneShotGuard(path); pin=h.matrix_pin()
    guard.claim(pin)
    assert b"IN_PROGRESS" in path.read_bytes()
    fail("one_shot",lambda:guard.claim(pin))
    fail("one_shot",lambda:h.OneShotGuard(path).claim(pin))
    guard.complete(pin,"1"*64)
    assert b"COMPLETE" in path.read_bytes()
    fail("one_shot",lambda:guard.claim(pin))
    fail("marker_mismatch",lambda:guard.complete(pin,"1"*64))


def test_actual_stdlib_child_isolated_and_never_native():
    result=h.subprocess_fake(request())
    assert result["status"]=="SYNTHETIC_ONLY"


def test_actual_child_timeout_killed_without_retry():
    fail("worker_timeout",lambda:h.subprocess_fake(request(),timeout=0))


@pytest.mark.parametrize("field,value",[("vector_blocks",True),("scalar_pixels",0.0),("float_lanes",4.0)])
def test_event_counts_are_exact_integer_metadata(field,value):
    r=request(); evidence=h.FakeProvider().evidence(r,"after")
    evidence["events"][field]=value
    fail("branch_mismatch",lambda:h.check_evidence(r,evidence,"after"))


def test_fake_matrix_marker_before_call_complete_and_rerun_refusal(tmp_path):
    guard=h.OneShotGuard(tmp_path/"matrix")
    calls=[]
    def transport(r):
        assert b"IN_PROGRESS" in guard.path.read_bytes()
        calls.append(h.request_hash(r))
        return fake_summary(r)
    result=h.run_fake_matrix(guard,transport)
    assert result["status"]=="SYNTHETIC_ONLY" and result["cases"]==492 and len(calls)==492
    assert b"COMPLETE" in guard.path.read_bytes()
    fail("one_shot",lambda:h.run_fake_matrix(guard,transport))
    assert len(calls)==492


def test_later_matrix_failure_consumed_marker_and_no_native_promotion(tmp_path):
    guard=h.OneShotGuard(tmp_path/"matrix")
    calls=[]
    def transport(r):
        calls.append(1)
        if len(calls)==2: return {"status":"PASS","request_sha":h.request_hash(r)}
        return fake_summary(r)
    fail("receipt_mismatch",lambda:h.run_fake_matrix(guard,transport))
    assert b"IN_PROGRESS" in guard.path.read_bytes() and b"COMPLETE" not in guard.path.read_bytes()
    fail("one_shot",lambda:h.run_fake_matrix(guard,transport))
    assert len(calls)==2


def test_native_gate_before_any_marker_creation(tmp_path):
    fail("unsupported_attestor",h.native_preflight)
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize("data",[b"["*13+b"]"*13,b"["+b"0,"*2049+b"0]",b"1"*65])
def test_decoder_preallocation_structure_gates(data,monkeypatch):
    monkeypatch.setattr(h.json,"loads",lambda *args:pytest.fail("JSON allocator before structure gate"))
    fail("metadata_cap",lambda:h.bounded_json(data))


def test_reference_file_pin_check_is_read_only_and_detects_drift(tmp_path,monkeypatch):
    path=tmp_path/"pin"; path.write_bytes(b"pinned")
    monkeypatch.setattr(h,"PINNED_FILES",(("pin",h.digest(b"pinned")),))
    assert h.reference_hashes(tmp_path)==h.PINNED_FILES
    path.write_bytes(b"changed")
    fail("reference_drift",lambda:h.reference_hashes(tmp_path))


def test_reference_root_scope_escape_refused(tmp_path,monkeypatch):
    monkeypatch.setattr(h,"PINNED_FILES",(("../outside","0"*64),))
    fail("scope_escape",lambda:h.reference_hashes(tmp_path))


def test_bad_float_source_with_zero_coefficients_before_backend():
    recipe=next(r for r in h.matrix_recipes() if r.kind=="f32le" and r.pattern=="integer")
    r=h.make_request(recipe)
    r=replace(r,source=replace(r.source,data=r.source.data[:4]+m.U32.pack(0x7FC00000)+r.source.data[8:]))
    provider=h.FakeProvider()
    fail("input_refusal",lambda:h.Worker(provider).run(r))
    assert provider.calls==0


def test_bad_native_offset_before_backend():
    r=request(); r=replace(r,map_y=replace(r.map_y,data=m.U32.pack(0x4F000000)* (len(r.map_y.data)//4)))
    provider=h.FakeProvider()
    fail("input_refusal",lambda:h.Worker(provider).run(r)); assert provider.calls==0


def test_oversized_direct_receipt_refuses_before_serialization(monkeypatch):
    r=request(); receipt=h.Worker(h.FakeProvider()).run(r)
    receipt["record"]["output_hex"]="0"*(2*h.parent_reservation(r)[3]+1)
    monkeypatch.setattr(h,"seal",lambda *args:pytest.fail("serialize bad receipt"))
    fail("output_mismatch",lambda:test_validate(r,receipt,require_child=False))


@pytest.mark.parametrize("target",["shape","pins"])
def test_decoder_rejects_sequence_substitution_before_tuple_expansion(target):
    value=json.loads(h.encode_request(request()))
    if target=="shape": value["source"]["shape"]="x"*4096
    else: value["kernel"]["source_files"]="x"*4096
    fail("wire_mismatch",lambda:h.decode_request(h.json_bytes(value)))


@pytest.mark.parametrize("field",["output_sha","request_sha","support_seal","outer_seal"])
@pytest.mark.parametrize("value",["a"*4096,"g"*64,"A"*64])
def test_unbounded_or_noncanonical_digest_refuses_before_serialization(field,value,monkeypatch):
    r=request(); receipt=h.Worker(h.FakeProvider()).run(r)
    if field=="outer_seal": receipt["seal"]=value
    else: receipt["record"][field]=value
    monkeypatch.setattr(h,"seal",lambda *args:pytest.fail("digest validation must precede seal allocation"))
    monkeypatch.setattr(h,"json_bytes",lambda *args:pytest.fail("digest validation must precede JSON allocation"))
    fail("receipt_mismatch",lambda:test_validate(r,receipt))


def test_extra_summary_fields_refuse_before_serializer(tmp_path,monkeypatch):
    guard=h.OneShotGuard(tmp_path/"marker")
    def transport(r):
        result=fake_summary(r); result["extra"]="x"*4096
        monkeypatch.setattr(h,"json_bytes",lambda *args:pytest.fail("invalid summary serialized"))
        return result
    fail("receipt_mismatch",lambda:h.run_fake_matrix(guard,transport))
    assert b"IN_PROGRESS" in guard.path.read_bytes()


def test_parent_reservation_accounts_fixed_receive_parse_allowance():
    r=request()
    size=h.parent_reservation(r)[3]
    total=sum(len(v.data) for v in (r.source,r.map_x,r.map_y))
    assert h.parent_reservation(r)[2]==8*total+8*size+3*(h.MAX_WIRE_BYTES+8)+h.PARENT_SCRATCH_BYTES
    assert h.parent_reservation(r)[2]<h.MAX_PARENT_BYTES
    assert isinstance(h.unframe(h.framed(b"{}")),memoryview)


def test_stale_same_request_receipt_from_distinct_attempt_refuses():
    r=request(); first="a"*64; second="b"*64
    receipt=h.Worker(h.FakeProvider()).run(r,attempt=first,expected_pid=os.getpid())
    fail("attempt_mismatch",lambda:h.validate_receipt(r,receipt,expected_attempt=second,
                                                      expected_pid=os.getpid(),require_child=False))


def test_other_child_pid_refuses_even_if_not_parent_and_resealed():
    r=request(); receipt=h.Worker(h.FakeProvider()).run(r)
    receipt["record"]["pid"]=os.getpid()+1
    receipt["seal"]=h.seal(receipt["record"])
    fail("isolation_mismatch",lambda:h.validate_receipt(r,receipt,expected_attempt=receipt["record"]["attempt"],
                                                        expected_pid=os.getpid()+2))


@pytest.mark.parametrize("phase",["before","after"])
@pytest.mark.parametrize("field,value",[("attempt","b"*64),("pid",1)])
def test_substituted_phase_attempt_or_pid_refuses_resealed(phase,field,value):
    r=request(); receipt=h.Worker(h.FakeProvider()).run(r,attempt="a"*64)
    receipt["record"][phase][field]=value
    receipt["seal"]=h.seal(receipt["record"])
    fail("attempt_mismatch",lambda:test_validate(r,receipt))


def test_reused_summary_attempt_refuses_consumed_plan(tmp_path):
    guard=h.OneShotGuard(tmp_path/"marker")
    def transport(r):
        result=fake_summary(r); result["attempt"]="a"*64; return result
    fail("attempt_mismatch",lambda:h.run_fake_matrix(guard,transport))
    assert b"COMPLETE" not in guard.path.read_bytes()


class FakePipe:
    def __init__(self,name,events,fault=None): self.name,self.events,self.fault=name,events,fault
    def close(self):
        self.events.append("close_"+self.name)
        if self.fault=="close_"+self.name: raise KeyboardInterrupt()


class FakeProcess:
    pid=98765
    def __init__(self,events,fault=None):
        self.events,self.fault,self.returncode=events,fault,None
        self.stdin=FakePipe("stdin",events,fault)
        self.stdout=FakePipe("stdout",events,fault)
        self.stderr=FakePipe("stderr",events,fault)
    def poll(self): return self.returncode
    def kill(self):
        self.events.append("kill")
        if self.fault=="kill": raise KeyboardInterrupt()
        self.returncode=-9
    def wait(self,timeout=None):
        self.events.append("reap")
        if self.fault=="reap": raise KeyboardInterrupt()
        self.returncode=-9
        return self.returncode


class FakeThread:
    def __init__(self,index,events,fault=None): self.index,self.events,self.fault,self.ident=index,events,fault,None
    def start(self):
        self.events.append("start_"+str(self.index)); self.ident=self.index+1
        if self.fault=="start_"+str(self.index): raise KeyboardInterrupt()
    def join(self,timeout=None):
        self.events.append("join_"+str(self.index))
        if self.fault=="join_"+str(self.index): raise KeyboardInterrupt()
    def is_alive(self): return self.fault=="alive_"+str(self.index)


@pytest.mark.parametrize("fault",["after_spawn","construct_0","construct_1","start_0","start_1"])
def test_post_spawn_setup_failure_always_attempts_owned_cleanup(fault,monkeypatch):
    events=[]; process=FakeProcess(events)
    monkeypatch.setattr(h.subprocess,"Popen",lambda *args,**kwargs:process)
    index=0
    def thread(*args,**kwargs):
        nonlocal index
        current=index; index+=1; events.append("construct_"+str(current))
        if fault=="construct_"+str(current): raise MemoryError()
        return FakeThread(current,events,fault)
    monkeypatch.setattr(h,"Thread",thread)
    if fault=="after_spawn":
        monkeypatch.setattr(h,"encode_attempt",lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises((MemoryError,KeyboardInterrupt)):
        h.subprocess_fake(request())
    assert "kill" in events and "reap" in events
    assert all("close_"+name in events for name in ("stdin","stdout","stderr"))
    if "start_0" in events: assert "join_0" in events
    if "start_1" in events: assert "join_1" in events
    assert process.poll() is not None


@pytest.mark.parametrize("fault",["kill","reap","close_stdin","close_stdout","close_stderr","join_0","join_1","alive_0","alive_1"])
def test_each_cleanup_failure_attempts_remaining_resources_and_latches_refusal(fault,monkeypatch):
    events=[]; process=FakeProcess(events,fault)
    monkeypatch.setattr(h.subprocess,"Popen",lambda *args,**kwargs:process)
    owner=h.ChildOwner("a"*64)
    fail("cleanup_incomplete",lambda:cleanup_owner(owner,events,fault))
    assert owner.phase=="incomplete"
    assert "kill" in events and "reap" in events
    assert all("close_"+name in events for name in ("stdin","stdout","stderr"))
    assert "join_0" in events and "join_1" in events
    fail("one_shot",lambda:owner.accept(request(),{}))


def cleanup_owner(owner,events,fault):
    with owner:
        owner.spawn()
        owner.threads=(FakeThread(0,events,fault),FakeThread(1,events,fault))
        for thread in owner.threads: thread.start()


def test_one_receipt_per_owned_attempt_and_actual_pid(monkeypatch):
    r=request(); receipt=h.Worker(h.FakeProvider()).run(r,attempt="a"*64)
    owner=h.ChildOwner("a"*64); owner.process=type("Process",(),{"pid":os.getpid()})()
    owner.phase="spawned"
    fail("isolation_mismatch",lambda:owner.accept(r,receipt))  # actual process must differ from parent
    fail("one_shot",lambda:owner.accept(r,receipt))


def test_worker_envelope_pid_is_actual_before_any_backend():
    provider=h.FakeProvider()
    fail("attempt_mismatch",lambda:h.Worker(provider).run(request(),attempt="a"*64,expected_pid=os.getpid()+1))
    assert provider.calls==0



def test_direct_same_python_identity_and_isolated_stdlib_launch(monkeypatch):
    events=[]; process=FakeProcess(events); captured=[]
    def spawn(args,**kwargs): captured.append(args); return process
    monkeypatch.setattr(h.subprocess,"Popen",spawn)
    owner=h.ChildOwner("a"*64)
    with owner: owner.spawn()
    assert captured[0][0]==str(h.REFERENCE_PYTHON.resolve())
    assert captured[0][1:4]==["-I","-S","-B"]
    assert h.worker_python()==str(h.REFERENCE_PYTHON.resolve())


def test_wrong_python_build_refuses_before_spawn(monkeypatch):
    monkeypatch.setattr(h,"REFERENCE_PYTHON_VERSION","other")
    monkeypatch.setattr(h.subprocess,"Popen",lambda *args,**kwargs:pytest.fail("wrong interpreter spawned"))
    fail("python_mismatch",lambda:h.subprocess_fake(request()))
