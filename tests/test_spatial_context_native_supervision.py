"""Synthetic temporary artifacts and fake OS API only. No process/Job calls."""
import ast
import copy
import inspect
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest
from scripts.evaluation import spatial_context_native_io as native
from scripts.evaluation import run_spatial_context_native as controller
from test_spatial_context_native_io import authored_manifest, manifest, seal, services, FakeArray


def observed_counters():
    def work(limit, processed):
        return dict(limit=limit,requested=processed,charged=processed,processed=processed,uncharged=0,unprocessed=0)
    arm=dict(comparisons=work(1048576,95090),tuples=work(4096,1),determinants=work(32768,8))
    return dict(footprint=dict(requested=16800,charged=16800,processed=16800,uncharged=0,unprocessed=0),
                arms=dict(A=copy.deepcopy(arm),B=copy.deepcopy(arm)))


def authored_auth(accepted):
    record=dict(schema="spatial-context-native-auth-v1",attempt_id=accepted.record["attempt_id"],
        manifest_file_sha256=accepted.file_sha256,command_sha256=native.sha(native.canonical(accepted.record["launch"])),
        review_receipts=[dict(reviewer=name,path=accepted.record["launch"]["cwd"]+"\\authored-"+name+"-review.json",sha256="a"*64)
                        for name in ("Luna","Astra")],single_use=True,disposition="AUTHORIZE_ONE_SPATIAL_CONTEXT_RUN")
    seal(record,"authorization_sha256")
    return native.canonical(record)+b"\n"


class FakeAPI:
    """Proof model only; no subprocess, ctypes, environment, native API or payload."""
    def __init__(self, accepted, auth, failure=None):
        self.manifest=accepted;self.auth=auth;self.failure=failure
        self.calls=[];self.finished=False;self.index=0
        self.channel=controller.Channel(accepted.record["attempt_id"])
        self.telemetry_changes={}
        self.reported_counters=observed_counters()
        self.candidate_change=None
        self.retained_output=[]

    def call(self,name):
        self.calls.append(name)
        if self.failure==name:raise native.Refusal("fake_api_failure")

    def private_channel(self,nonce):
        self.call("private_channel");self.nonce=nonce;return object()
    def create_job(self,limits):
        self.call("create_job")
        assert limits==dict(committed_bytes=2147483648,kill_on_close=True,no_breakaway=True,descendant_containment=True)
        return object()
    def create_suspended(self,launch,private):
        self.call("create_suspended")
        assert Path(self.manifest.record["artifacts"]["consumed_path"]).exists()
        assert set(launch["environment"])==set(native.ENVIRONMENT_KEYS)
        return object()
    def attach(self,job,child):self.call("attach")
    def enforcement(self,job,child):
        self.call("enforcement")
        proof=dict(suspended=True,attached=True,committed_bytes=2147483648,kill_on_close=True,
            no_breakaway=True,descendant_containment=True,private_inherited=True)
        if self.failure=="unproven":proof["no_breakaway"]=False
        return proof
    def resume(self,child):
        self.call("resume")
        assert self.calls.index("attach")<self.calls.index("enforcement")<self.calls.index("resume")
    def telemetry(self,job):
        self.call("telemetry")
        changes=self.telemetry_changes if self.finished else {}
        return {"status":"running","committed_bytes":1024,"peak_committed_bytes":2048,
                "descendants":0 if self.finished else 1,**changes}
    def send(self,private,frame):
        self.call("send")
        kind,payload=self.channel.decode(frame)
        assert kind in ("ack","telemetry")
    def next_event(self,child,max_bytes):
        self.call("next_event")
        assert max_bytes==4100
        if self.retained_output:
            return self.retained_output.pop(0)
        index=self.index;self.index+=1
        if index==0:
            payload=dict(manifest_file_sha256=self.manifest.file_sha256,authorization_file_sha256=native.sha(self.auth),nonce=self.nonce)
            if self.failure=="nonce":payload["nonce"]="f"*64
            return "control",self.channel.encode("hello",payload)
        if index<=len(controller.STAGES):
            stage=controller.STAGES[index-1]
            if self.failure=="stage_skip" and index==1:stage="preaccess"
            return "control",self.channel.encode("checkpoint",dict(stage=stage))
        if index==len(controller.STAGES)+1:
            artifacts=self.manifest.record["artifacts"]
            evidence=b"authored-synthetic-evidence"
            Path(artifacts["evidence"]).write_bytes(evidence)
            candidate_record=dict(status="complete",arm_dispositions=["complete_witness","complete_no_witness"],
                row176=dict(status="untestable_footprint",sites=0),attempt_id=self.manifest.record["attempt_id"],
                manifest_file_sha256=self.manifest.file_sha256,authorization_file_sha256=native.sha(self.auth),
                runtime_attestation_sha256=self.manifest.record["runtime"]["attestation"]["sha256"],evidence_sha256=native.sha(evidence))
            if self.candidate_change is not None:
                self.candidate_change(candidate_record)
            candidate=native.canonical(candidate_record)+b"\n"
            Path(artifacts["worker_candidate"]).write_bytes(candidate)
            self.candidate=candidate
            return "control",self.channel.encode("terminal",dict(status="staged_complete",code="none",worker_exit_intent=0,
                evidence=dict(bytes=len(evidence),sha256=native.sha(evidence)),candidate=dict(bytes=len(candidate),sha256=native.sha(candidate)),counters=self.reported_counters))
        self.finished=True
        if self.failure=="log":return "stdout",b"x"*65537
        if self.failure=="bad_exit":return "exit",1
        if self.failure=="candidate_drift":Path(self.manifest.record["artifacts"]["worker_candidate"]).write_bytes(b"changed")
        return "exit",0
    def terminate(self,job):self.call("terminate")
    def close(self,job):self.call("close")


def setup(tmp_path, failure=None):
    record,_,_=authored_manifest(str(tmp_path))
    accepted=manifest(record);auth=authored_auth(accepted)
    api=FakeAPI(accepted,auth,failure)
    return accepted,auth,api


def test_fake_suspended_attach_resume_and_authoritative_after_exit(tmp_path):
    accepted,auth,api=setup(tmp_path)
    result=controller.Controller(accepted,api,clock=lambda:1).run(auth)
    assert result==dict(status="complete",completion=True)
    assert api.finished and api.calls.index("attach")<api.calls.index("resume")
    artifacts=accepted.record["artifacts"]
    assert Path(artifacts["completion"]).exists() and not Path(artifacts["worker_candidate"]).exists()
    assert Path(artifacts["evidence"]).read_bytes()==b"authored-synthetic-evidence"
    assert json.loads(Path(artifacts["completion"]).read_bytes())["worker_exit"]==0
    assert api.calls[-1]=="close"


@pytest.mark.parametrize("failure",["private_channel","create_job","create_suspended","attach","enforcement","unproven",
    "resume","telemetry","send","next_event","nonce","stage_skip","bad_exit","candidate_drift","log"])
def test_every_fake_controller_boundary_retains_no_false_completion(tmp_path,failure):
    accepted,auth,api=setup(tmp_path,failure)
    result=controller.Controller(accepted,api,clock=lambda:1).run(auth)
    assert result["status"]=="incomplete" and not result["completion"]
    assert result["counter_tail"]==("observed_terminal" if failure in ("bad_exit","candidate_drift","log") else "unknown")
    assert result["sampled_rss"] is None
    assert not Path(accepted.record["artifacts"]["completion"]).exists()
    if failure in ("attach","enforcement","unproven"):
        assert "resume" not in api.calls


@pytest.mark.parametrize("changes",[dict(status="terminated"),dict(committed_bytes=2147483649),dict(peak_committed_bytes=2147483649),dict(descendants=1)])
def test_late_job_telemetry_with_same_terminal_refuses(tmp_path,changes):
    accepted,auth,api=setup(tmp_path);api.telemetry_changes=changes
    result=controller.Controller(accepted,api,clock=lambda:1).run(auth)
    assert not result["completion"] and not Path(accepted.record["artifacts"]["completion"]).exists()
    assert Path(accepted.record["artifacts"]["evidence"]).exists()


def test_single_deadline_includes_launch_and_finalization(tmp_path):
    accepted,auth,api=setup(tmp_path)
    clock=lambda:181 if api.finished else 0
    result=controller.Controller(accepted,api,clock=clock).run(auth)
    assert not result["completion"] and Path(accepted.record["artifacts"]["evidence"]).exists()


def test_no_clobber_consumption_and_second_attempt_never_launches(tmp_path):
    accepted,auth,api=setup(tmp_path)
    Path(accepted.record["artifacts"]["consumed_path"]).write_bytes(b"authored-existing-marker")
    with pytest.raises(FileExistsError):controller.Controller(accepted,api,clock=lambda:0).run(auth)
    assert api.calls==[]


@pytest.mark.parametrize("kind",["self_seal","manifest_hash","command_hash","review_order","single_use","extra"])
def test_auth_non_circular_binding_and_closed_rejection(tmp_path,kind):
    accepted,auth,api=setup(tmp_path);record=json.loads(auth)
    if kind=="self_seal":record["authorization_sha256"]="f"*64
    elif kind=="manifest_hash":record["manifest_file_sha256"]="f"*64
    elif kind=="command_hash":record["command_sha256"]="f"*64
    elif kind=="review_order":record["review_receipts"].reverse()
    elif kind=="single_use":record["single_use"]=False
    else:record["extra"]="bad"
    with pytest.raises(native.Refusal):controller.authorize(native.canonical(record)+b"\n",accepted,native.sha(native.canonical(accepted.record["launch"])))
    assert api.calls==[]


def test_private_message_round_trip_sequence_replay_and_combined_budget():
    budget=controller.ControlBudget();a=controller.Channel("authored",budget);b=controller.Channel("authored",budget)
    frame=a.encode("checkpoint",dict(stage="worker_bootstrap"))
    assert b.decode(frame)==("checkpoint",dict(stage="worker_bootstrap"))
    with pytest.raises(native.Refusal):b.decode(frame)
    budget.bytes=65536
    with pytest.raises(native.Refusal):a.encode("checkpoint",dict(stage="preaccess"))


@pytest.mark.parametrize("frame",[b"",struct.pack("<I",4097),struct.pack("<I",0),struct.pack("<I",2)+b"{}x",
    struct.pack("<I",13)+b'{"x":1,"x":2}',struct.pack("<I",13)+b'{"x":NaNxxxx}'])
def test_message_length_duplicate_trailing_nonfinite_refusal(frame):
    with pytest.raises(native.Refusal):controller.Channel("authored").decode(frame)


def test_checkpoint_no_skip_and_stale_or_over_cap_before_loader():
    calls=[]
    def exchange(kind,payload):
        calls.append(payload["stage"])
        return dict(stage="preaccess",status="running",elapsed_ms=0,committed_bytes=0,log_bytes=0)
    supervisor=controller.WorkerSupervisor(exchange)
    with pytest.raises(native.Refusal,match="stale_telemetry"):supervisor.checkpoint("worker_bootstrap")
    with pytest.raises(native.Refusal):supervisor.checkpoint("decode_0")
    assert calls==["worker_bootstrap"]


@pytest.mark.parametrize("proof",[None,SimpleNamespace(authenticated=False,inherited=True,job_enforced=True),
    SimpleNamespace(authenticated=True,inherited=False,job_enforced=True),SimpleNamespace(authenticated=True,inherited=True,job_enforced=False)])
def test_missing_private_or_job_proof_zero_activation(proof):
    with pytest.raises(native.Refusal):controller.WorkerGrant(proof,"a"*64)


def test_grant_one_shot_replay_and_worker_invalid_zero_loader():
    grant=controller.WorkerGrant(SimpleNamespace(authenticated=True,inherited=True,job_enforced=True),"a"*64)
    assert grant.valid("a"*64) and grant.consume("a"*64) and not grant.consume("a"*64)
    calls=[]
    assert controller.worker(None,grant,None,lambda *args:calls.append(True))==1
    assert calls==[]


@pytest.mark.parametrize("failure",["link","after_link"])
def test_atomic_completion_failure_removes_only_our_marker(tmp_path,monkeypatch,failure):
    path=tmp_path/"completion.json";other=tmp_path/"authored-unrelated.bin";other.write_bytes(b"preserved")
    if failure=="link":monkeypatch.setattr(controller.os,"link",lambda *a:(_ for _ in ()).throw(OSError("fake-link")))
    count=0
    def validate():
        nonlocal count
        count+=1
        if count==2:raise native.Refusal("late_resource_limit")
    with pytest.raises((native.Refusal,OSError)):controller.exclusive_bytes(path,b"authored-synthetic",65536,final=True,validate=validate)
    assert not path.exists() and other.read_bytes()==b"preserved"
    assert not path.with_name(".completion.json.partial").exists()


def test_ledger_counts_aliases_and_cap_plus_one(tmp_path):
    path=tmp_path/"authored-evidence.json";path.write_bytes(b"1234")
    controller.os.link(path,tmp_path/"authored-alias.json")
    ledger=controller.Ledger(tmp_path)
    assert ledger.current()==8
    ledger.precharge(native.LIMITS["publication_bytes"]-8)
    with pytest.raises(native.Refusal):ledger.precharge(native.LIMITS["publication_bytes"]-7)


def test_modules_are_stdlib_only_no_loader_launch_pickle_or_environment_at_import():
    for module in (native,controller):
        tree=ast.parse(inspect.getsource(module))
        imports=[]
        for node in tree.body:
            if isinstance(node,ast.Import):imports.extend(alias.name for alias in node.names)
            elif isinstance(node,ast.ImportFrom):imports.append(node.module)
        assert not any(name.startswith(("numpy","PIL","cv2","mc_photogrammetry","scripts.evaluation")) for name in imports)
        assert not any(isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) for node in tree.body)
        assert "pickle" not in imports and "subprocess" not in imports and "ctypes" not in imports
    with pytest.raises(native.Refusal,match="os_api_required"):controller.main([r"C:\authored\manifest.json","a"*64,r"C:\authored\auth.json"])


@pytest.mark.parametrize("kind",["unknown","missing","status","inconclusive_arm","one_arm","sentinel_bool","sentinel_status","digest"])
def test_closed_candidate_refuses_before_parent_completion(tmp_path,kind):
    accepted,auth,api=setup(tmp_path)
    def mutate(record):
        if kind=="unknown":record["extra"]="bad"
        elif kind=="missing":record.pop("row176")
        elif kind=="status":record["status"]="incomplete"
        elif kind=="inconclusive_arm":record["arm_dispositions"][1]="budget_inconclusive"
        elif kind=="one_arm":record["arm_dispositions"].pop()
        elif kind=="sentinel_bool":record["row176"]["sites"]=False
        elif kind=="sentinel_status":record["row176"]["status"]="complete"
        else:record["evidence_sha256"]="f"*64
    api.candidate_change=mutate
    result=controller.Controller(accepted,api,clock=lambda:1).run(auth)
    assert not result["completion"] and not Path(accepted.record["artifacts"]["completion"]).exists()
    failure=json.loads(Path(accepted.record["artifacts"]["failure"]).read_bytes())
    controller.failure_record(failure)
    assert failure["counters"]==observed_counters() and failure["counter_tail"]=="observed_terminal"


@pytest.mark.parametrize("kind",["unavailable","missing_arm","partial_footprint","oversized_counter","remainder","arm_remainder","regression"])
def test_final_counters_complete_and_monotonic_guards(tmp_path,kind):
    accepted,auth,api=setup(tmp_path)
    if kind=="unavailable":api.reported_counters=None
    elif kind=="missing_arm":api.reported_counters["arms"]["B"]=None
    elif kind=="partial_footprint":api.reported_counters["footprint"].update(processed=16000,unprocessed=800)
    elif kind=="oversized_counter":api.reported_counters["footprint"].update(requested=16801,uncharged=1)
    elif kind=="remainder":api.reported_counters["footprint"].update(requested=16800,charged=16799,processed=16799,uncharged=1,unprocessed=0)
    elif kind=="arm_remainder":api.reported_counters["arms"]["A"]["comparisons"].update(requested=95091,uncharged=1)
    else:
        before=observed_counters();after=observed_counters()
        after["arms"]["A"]["comparisons"].update(requested=1,charged=1,processed=1)
        with pytest.raises(native.Refusal,match="counter_regression"):controller.monotonic_counters(before,after)
        return
    result=controller.Controller(accepted,api,clock=lambda:1).run(auth)
    assert not result["completion"] and not Path(accepted.record["artifacts"]["completion"]).exists()
    assert (result["counters"] is None)==(kind in ("unavailable","oversized_counter"))
    if kind in ("oversized_counter","remainder"):
        failure=json.loads(Path(accepted.record["artifacts"]["failure"]).read_bytes())
        controller.failure_record(failure)
        assert failure["publication_state"]["completion_present"] is False
        if kind=="oversized_counter":
            assert result["counters"] is None and failure["counters"] is None
            assert result["counter_tail"]==failure["counter_tail"]=="unknown"
        else:
            assert result["counters"]==failure["counters"]==api.reported_counters
            assert result["counter_tail"]==failure["counter_tail"]=="observed_terminal"
            controller.counters(result["counters"],stage=False)
            with pytest.raises(native.Refusal,match="complete_counts_unavailable"):
                controller.completed_counters(result["counters"])


@pytest.mark.parametrize("cleanup",["terminate","close","both"])
def test_durable_log_failure_and_cleanup_errors_preserve_counters(tmp_path,cleanup):
    accepted,auth,api=setup(tmp_path,"bad_exit")
    api.retained_output=[("stdout",b"authored stdout\n"),("stderr",b"authored stderr\n")]
    def fail(name):
        api.calls.append(name)
        if cleanup in (name,"both"):raise RuntimeError("authored exception text must never escape or persist")
    api.terminate=lambda job:fail("terminate")
    api.close=lambda job:fail("close")
    result=controller.Controller(accepted,api,clock=lambda:1).run(auth)
    artifacts=accepted.record["artifacts"]
    assert not result["completion"] and not Path(artifacts["completion"]).exists()
    assert Path(artifacts["log"]).read_bytes()==b"authored stdout\nauthored stderr\n"
    wire=Path(artifacts["failure"]).read_bytes();record=json.loads(wire)
    controller.failure_record(record)
    assert record["counters"]==observed_counters() and record["counter_tail"]=="observed_terminal"
    assert record["cleanup_failures"]==(["terminate","close"] if cleanup=="both" else [cleanup])
    assert b"authored exception text" not in wire and len(wire)<=65536
    controller.Ledger(Path(artifacts["output_directory"])).small_aliases(0)


def test_close_failure_after_commit_rolls_back_authority_and_writes_failure(tmp_path):
    accepted,auth,api=setup(tmp_path)
    def close(job):api.calls.append("close");raise OSError("authored close failure")
    api.close=close
    result=controller.Controller(accepted,api,clock=lambda:1).run(auth)
    assert not result["completion"] and not Path(accepted.record["artifacts"]["completion"]).exists()
    assert Path(accepted.record["artifacts"]["failure"]).exists()


def test_small_alias_budget_and_retirement_before_final(tmp_path,monkeypatch):
    accepted,auth,api=setup(tmp_path);original=controller.exclusive_bytes;seen=[]
    def observe(path,*args,**kwargs):
        if Path(path).name=="completion.json":
            assert not Path(accepted.record["artifacts"]["worker_candidate"]).exists()
            assert not Path(path).with_name(".completion.json.partial").exists()
            assert Path(accepted.record["artifacts"]["log"]).exists()
            seen.append(True)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(controller,"exclusive_bytes",observe)
    assert controller.Controller(accepted,api,clock=lambda:1).run(auth)["completion"] and seen==[True]
    ledger=controller.Ledger(Path(accepted.record["artifacts"]["output_directory"]))
    with pytest.raises(native.Refusal,match="small_alias_budget"):ledger.small_aliases(2)


def fake_worker_fixture(tmp_path,monkeypatch,failure=None):
    """Authored driver contract double; no accepted coordinator/native import."""
    record,diagnostic,png=authored_manifest(str(tmp_path));accepted=manifest(record)
    output=Path(record["artifacts"]["output_directory"]);output.mkdir()
    fake=services();events=[];prepared=[];frames=[];terminal=[]
    def boundary(name):
        events.append(name)
        if failure==name:raise native.Refusal("authored_worker_failure")
    class MemoryReaders:
        def __init__(self,manifest,grant):assert grant.active;self.used=set()
        def read_payload(self,kind,key,limit):
            boundary("read_"+key)
            assert (kind,key) not in self.used;self.used.add((kind,key))
            wire=diagnostic if kind=="diagnostic" else png
            assert limit==len(wire)+1
            return wire
    monkeypatch.setattr(native,"FixedReaders",MemoryReaders)
    original_decode=fake.load_image_bytes
    def decode(data,**kwargs):boundary("rgb");return original_decode(data,**kwargs)
    fake.load_image_bytes=decode
    if failure=="ingress":fake.image_sha256=lambda pixels:"f"*64
    original_mask=fake.diagnostic._decode_source_mask
    def mask(entry):boundary("mask_"+str(entry["reading_index"]));return original_mask(entry)
    fake.diagnostic._decode_source_mask=mask
    def prepare(pixels,mask,usable,*,role):
        index=len(prepared);boundary("prepare_"+str(index))
        receipt=SimpleNamespace(role=role,source_refs=(pixels,mask,usable),plan=SimpleNamespace(eligible_count=(0,37,2570)[index]))
        prepared.append(receipt);return receipt
    def extract(receipt):
        boundary("extract_"+str(len(frames)))
        assert receipt in prepared[1:] and receipt is not prepared[0]
        frame=SimpleNamespace(receipt=receipt,source_refs=receipt.source_refs,tiles=object());frames.append(frame);return frame
    def match(arm,left,right):
        boundary("match_"+arm)
        assert left is frames[0] and right is frames[1]
        return copy.deepcopy(observed_counters()["arms"][arm])
    def publish(path,record):
        boundary("evidence")
        wire=native.canonical(record);Path(path).write_bytes(wire)
        return dict(bytes=len(wire),sha256=native.sha(wire),aggregate_peak=len(wire))
    names=("capability","in_memory_inventory","source_metadata","read_payload","inspect_diagnostic","decode_rgb",
           "ingress_digest","decode_mask","usable_mask","prepare_shared_frame","extract_prepared_frame","match","supervisor","publication")
    driver=fake.driver
    driver.Request=lambda **kwargs:SimpleNamespace(**kwargs)
    driver.Limits=lambda **kwargs:SimpleNamespace(**kwargs)
    driver.Dependencies=lambda *args:SimpleNamespace(**dict(zip(names,args)))
    driver.codec=SimpleNamespace(canonical=native.canonical,sha=native.sha,dumps=native.canonical)
    driver.validate_request=lambda request:None
    def coordinate(request,deps):
        result=SimpleNamespace(status="incomplete",completion=False,counters={},evidence_receipt=None)
        try:
            assert deps.capability.valid(request.bindings["authorization_sha256"])
            assert deps.capability.consume(request.bindings["authorization_sha256"])
            deps.supervisor.checkpoint("preaccess")
            assert deps.source_metadata() is request.metadata
            wire=deps.read_payload("diagnostic","sealed_source",request.diagnostic_length+1)
            boundary("diagnostic");deps.inspect_diagnostic(wire)
            arrays=[]
            for index,source in enumerate(request.bindings["sources"]):
                deps.supervisor.checkpoint("decode_"+str(index))
                encoded=deps.read_payload("encoded",source["frame_id"],source["encoded_length"]+1)
                pixels=deps.decode_rgb(encoded,source["frame_id"])
                assert deps.ingress_digest(pixels)==source["ingress_sha256"]
                mask=deps.decode_mask(wire,source["region_id"])
                usable=deps.usable_mask(pixels,source["usable_helper_sha256"])
                arrays.append((pixels,mask,usable))
            deps.supervisor.checkpoint("row176_mask")
            sentinel=(arrays[0][0],deps.decode_mask(wire,request.row176["region_id"]),arrays[0][2])
            result.counters["footprint"]=dict(requested=16800,charged=16800,processed=0,uncharged=0,unprocessed=16800)
            for index,(label,items,role) in enumerate(zip(native.LABELS,[sentinel,*arrays],(0,0,1))):
                deps.supervisor.checkpoint("prepare_"+label)
                deps.prepare_shared_frame(prepare,*items,role=role)
                result.counters["footprint"]["processed"]+=5600;result.counters["footprint"]["unprocessed"]-=5600
            assert tuple(p.plan.eligible_count for p in prepared)==(0,37,2570)
            result.counters["arms"]={}
            for index,receipt in enumerate(prepared[1:]):
                deps.supervisor.checkpoint("extract_"+str(index));deps.extract_prepared_frame(extract,receipt)
            for name in ("A","B"):
                deps.supervisor.checkpoint("match_"+name)
                result.counters["arms"][name]=deps.match(match,name,*frames)
            deps.supervisor.checkpoint("publication")
            evidence_record=dict(authored="synthetic-evidence",observed=result.counters)
            result.evidence_receipt=deps.publication.evidence(publish,evidence_record,request.limits)
            assert deps.publication.readback(result.evidence_receipt["bytes"]+1)==native.canonical(evidence_record)
            for stage in ("completion","completion_commit","terminal_commit"):deps.supervisor.checkpoint(stage)
            boundary("candidate")
            completion=native.canonical(dict(status="complete",evidence_sha256=result.evidence_receipt["sha256"],
                arm_dispositions=["complete_witness","complete_no_witness"],row176=dict(status="untestable_footprint",sites=0)))
            assert deps.publication.complete(completion,request.limits)
            result.completion=True;result.status="complete"
        except native.Refusal:
            pass
        return result
    driver.coordinate=lambda request,deps:deps.supervisor.run(lambda:coordinate(request,deps),request.limits)
    def exchange(kind,payload):
        if kind=="terminal":terminal.append(payload);return None
        assert kind=="checkpoint"
        events.append("checkpoint_"+payload["stage"])
        return dict(stage=payload["stage"],status="running",elapsed_ms=0,committed_bytes=1024,log_bytes=0)
    grant=controller.WorkerGrant(SimpleNamespace(authenticated=True,inherited=True,job_enforced=True),"a"*64)
    return accepted,grant,fake,events,prepared,frames,terminal,exchange


def test_successful_fake_worker_wires_all_adapters_once_without_public_completion(tmp_path,monkeypatch):
    accepted,grant,fake,events,prepared,frames,terminal,exchange=fake_worker_fixture(tmp_path,monkeypatch)
    assert controller.worker(accepted,grant,exchange,lambda manifest,capability:fake)==0
    assert len(terminal)==1 and terminal[0]["status"]=="staged_complete"
    assert terminal[0]["counters"]==observed_counters()
    assert [call for call in fake.diagnostic.calls if type(call) is tuple]==[("unpack",1),("unpack",2),("unpack",0)]
    assert len([event for event in fake.events if event[0]=="decode"])==2
    assert tuple(item.plan.eligible_count for item in prepared)==(0,37,2570) and len(frames)==2
    assert prepared[0].source_refs[0] is prepared[1].source_refs[0] and prepared[0].source_refs[2] is prepared[1].source_refs[2]
    assert events.index("match_A")<events.index("match_B")<events.index("evidence")<events.index("candidate")
    artifacts=accepted.record["artifacts"]
    assert Path(artifacts["worker_candidate"]).exists() and not Path(artifacts["completion"]).exists()
    controller.staged_candidate(json.loads(Path(artifacts["worker_candidate"]).read_bytes()))


@pytest.mark.parametrize("failure",["diagnostic","rgb","ingress","mask_1","mask_2","mask_0","prepare_0","prepare_1","prepare_2",
    "extract_0","extract_1","match_A","match_B","evidence","candidate"])
def test_fake_worker_failure_retains_available_final_work_and_single_terminal(tmp_path,monkeypatch,failure):
    accepted,grant,fake,events,prepared,frames,terminal,exchange=fake_worker_fixture(tmp_path,monkeypatch,failure)
    assert controller.worker(accepted,grant,exchange,lambda manifest,capability:fake)==1
    assert len(terminal)==1 and terminal[0]["status"]=="incomplete" and terminal[0]["candidate"] is None
    assert not Path(accepted.record["artifacts"]["completion"]).exists()
    if failure in ("diagnostic","rgb","ingress","mask_1","mask_2","mask_0"):
        assert terminal[0]["counters"] is None
    else:
        assert terminal[0]["counters"]["footprint"] is not None
        controller.counters(terminal[0]["counters"],stage=False)
    if failure=="diagnostic":assert "rgb" not in events and not any(type(call) is tuple for call in fake.diagnostic.calls)
    if failure=="match_B":
        assert terminal[0]["counters"]["arms"]["A"] is not None and terminal[0]["counters"]["arms"]["B"] is None


def test_acknowledged_partial_counters_survive_crash_without_terminal(tmp_path):
    accepted,auth,api=setup(tmp_path);original=api.next_event;sent=False
    available=observed_counters();available["arms"]["B"]=None
    def next_event(child,max_bytes):
        nonlocal sent
        if api.index==len(controller.STAGES)+1:
            if sent:raise OSError("authored abrupt worker crash")
            sent=True
            return "control",api.channel.encode("counters",dict(stage=controller.STAGES[-1],**available))
        return original(child,max_bytes)
    api.next_event=next_event
    result=controller.Controller(accepted,api,clock=lambda:1).run(auth)
    assert not result["completion"] and result["counter_tail"]=="unknown" and result["counters"]==available
    failure=json.loads(Path(accepted.record["artifacts"]["failure"]).read_bytes())
    assert failure["counters"]==available and failure["counter_tail"]=="unknown"


@pytest.mark.parametrize("status,code,intent",[("staged_complete","refused",0),("incomplete","none",1),
    ("inconclusive","child_failure",2),("staged_complete","none",True)])
def test_typed_terminal_refuses_status_code_exit_contradictions(status,code,intent):
    candidate=dict(bytes=1,sha256="a"*64) if status=="staged_complete" else None
    with pytest.raises(native.Refusal):
        controller.message_shape("terminal",dict(status=status,code=code,worker_exit_intent=intent,
            evidence=None,candidate=candidate,counters=None))


def test_failure_code_never_calls_exception_string_or_persists_arbitrary_text():
    class AuthoredError(native.Refusal):
        def __str__(self):raise AssertionError("arbitrary exception string accessed")
    assert controller.failure_code(AuthoredError({"unbounded":"authored"}))=="refused"
    assert controller.failure_code(AuthoredError("x"*100000))=="refused"


@pytest.mark.parametrize("failure",["temporary","temporary_and_completion","temporary_completion_and_log"])
def test_final_temporary_unlink_rollback_and_saturated_failure_publication(tmp_path,monkeypatch,failure):
    accepted,auth,api=setup(tmp_path)
    api.retained_output=[("stdout",b"authored retained log\n")]
    artifacts=accepted.record["artifacts"];original_unlink=Path.unlink
    other=tmp_path/"authored-unrelated.txt";other.write_bytes(b"preserve unrelated")
    calls=[]
    def unlink(path,*args,**kwargs):
        calls.append(path.name)
        if path.name==".completion.json.partial":raise OSError("authored temporary unlink failure")
        if failure!="temporary" and path.name=="completion.json":raise OSError("authored rollback unlink failure")
        if failure=="temporary_completion_and_log" and path.name=="worker.log":raise OSError("authored log retirement failure")
        return original_unlink(path,*args,**kwargs)
    monkeypatch.setattr(Path,"unlink",unlink)
    result=controller.Controller(accepted,api,clock=lambda:1).run(auth)
    assert not result["completion"] and other.read_bytes()==b"preserve unrelated"
    expected=dict(completion_present=failure!="temporary",temporary_present=True,
        log_present=failure!="temporary_and_completion",failure_present=failure!="temporary_completion_and_log")
    assert result["publication_state"]==expected
    ledger=controller.Ledger(Path(artifacts["output_directory"]));ledger.small_aliases(0)
    assert result["log"]["sha256"]==native.sha(b"authored retained log\n")
    assert result["log"]["retired"]==(failure=="temporary_and_completion")
    if failure=="temporary_completion_and_log":
        assert result["code"]=="failure_publication_unavailable" and result["failure_publication"]=="unavailable"
        assert not Path(artifacts["failure"]).exists()
        assert sum(path.name in ("completion.json",".completion.json.partial","worker.log")
                   for path in Path(artifacts["output_directory"]).iterdir())==3
    else:
        assert result["failure_publication"]=="published"
        record=json.loads(Path(artifacts["failure"]).read_bytes());controller.failure_record(record)
        assert record["publication_state"]==expected and record["log"]==result["log"]
        assert record["code"]=="publication_failure" and record["status"]=="incomplete"
    assert ".completion.json.partial" in calls and "completion.json" in calls


def test_transactional_temporary_cleanup_refuses_without_leaking_arbitrary_exception(tmp_path,monkeypatch):
    path=tmp_path/"completion.json";temporary=path.with_name(".completion.json.partial");original=Path.unlink
    def unlink(target,*args,**kwargs):
        if target==temporary:raise RuntimeError("authored arbitrary cleanup exception")
        return original(target,*args,**kwargs)
    monkeypatch.setattr(Path,"unlink",unlink)
    with pytest.raises(native.Refusal,match="publication_cleanup_failure"):
        controller.exclusive_bytes(path,b"authored completion",65536,final=True)
    assert not path.exists() and temporary.read_bytes()==b"authored completion"
