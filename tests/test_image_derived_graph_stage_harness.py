from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from PIL import Image


ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/"scripts/run_image_derived_graph_stage.py"
SPEC=importlib.util.spec_from_file_location("graph_stage_harness",PATH)
subject=importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name]=subject; SPEC.loader.exec_module(subject)


@dataclass
class Row:
    frame_id:str; canonical_index:int; alternative_index:int; signature_sha256:str


class Record:
    def __init__(self, value, digest=""): self.value=value; self.result_sha256=digest
    def to_record(self): return self.value


def _real_unavailable_graph():
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    return graph_mod.ImageDerivedGraphResult(
        graph_mod.IMAGE_DERIVED_GRAPH_SCHEMA,graph_mod.IMAGE_DERIVED_GRAPH_PRODUCER,
        graph_mod.GRAPH_TERMINAL,"unavailable","insufficient_axis_coverage","",
        "1"*64,"2"*64,"3"*64,("image-0000",),(),(),(),None)


def _inputs(tmp_path,count=4):
    paths=[]; hashes=[]
    for i in range(count):
        path=tmp_path/f"image-{i:04d}.png"; Image.new("RGB",(8,6),(i,2,3)).save(path)
        paths.append(path); hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    return paths,hashes


def _manifest(tmp_path, output, deadline):
    bound=[]
    for index in range(4):
        path=tmp_path/f"bound-{index}.png"; path.write_bytes(b"bound"+bytes([index]))
        bound.append({"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    value={"schema":"mc-photogrammetry-evaluation-graph-stage-manifest-v1",
        "pngs":bound,"producers":[{"path":str(PATH),"sha256":hashlib.sha256(PATH.read_bytes()).hexdigest()}],
        "limit_bytes":8*1024**3,"deadline_seconds":deadline,
        "output_path":str(output),"project_path":str(tmp_path/"project.mcscan")}
    raw=json.dumps(value,sort_keys=True,separators=(",",":")).encode(); value["manifest_sha256"]=hashlib.sha256(raw).hexdigest()
    path=tmp_path/"manifest.json"; path.write_text(json.dumps(value)); return path


def _api(calls):
    class Entry:
        class Ready: status="ok"
        session=object(); pose_graph=Ready(); graph=Ready()
    class Signatures:
        status="assembled"
        signatures=(Row("image-0000",1,2,"a"*64),)
    def entry(paths,project_path):
        calls.append(("entry",paths)); Path(project_path).write_text('{"schema":2}')
        return Entry()
    class Lattice: status="assembled"
    def lattice(**kwargs): calls.append(("lattice",kwargs)); return Lattice()
    def signatures(**kwargs): calls.append(("signatures",kwargs)); return Signatures()
    def graph(**kwargs):
        calls.append(("graph",kwargs)); return _real_unavailable_graph(),Record({"axis":[]},"c"*64)
    return subject.StageApi(entry,lattice,signatures,graph)


def test_sink_off_graph_digest_parity_and_zero_replay_calls(tmp_path):
    paths,hashes=_inputs(tmp_path); calls=[]
    first=subject.execute_graph_stage(png_paths=paths,expected_sha256=hashes,
        project_path=tmp_path/"a.mcscan",output_path=tmp_path/"a.json",
        telemetry_path=tmp_path/"a-heartbeat.json",api=_api(calls))
    second=subject.execute_graph_stage(png_paths=paths,expected_sha256=hashes,
        project_path=tmp_path/"b.mcscan",output_path=tmp_path/"b.json",
        telemetry_path=tmp_path/"b-heartbeat.json",api=_api([]))
    assert first["graph"]==second["graph"] and first["trace"]==second["trace"]
    graph_kwargs=next(value for name,value in calls if name=="graph")
    assert graph_kwargs["matcher_witness_sink"] is None
    assert graph_kwargs["matcher_witness_payload_sink"] is None
    assert graph_kwargs["witness_budget"] is None
    assert first["signature_identities"][0]["signature_sha256"]=="a"*64


def test_tampered_png_refuses_before_any_stage(tmp_path):
    paths,hashes=_inputs(tmp_path); paths[0].write_bytes(paths[0].read_bytes()+b"x"); calls=[]
    with pytest.raises(ValueError,match="provenance mismatch"):
        subject.execute_graph_stage(png_paths=paths,expected_sha256=hashes,
            project_path=tmp_path/"x.mcscan",output_path=tmp_path/"x.json",
            telemetry_path=tmp_path/"x-heartbeat.json",api=_api(calls))
    assert calls==[]


def test_telemetry_is_nonaccepting_and_final_publication_atomic(tmp_path):
    paths,hashes=_inputs(tmp_path)
    result=subject.execute_graph_stage(png_paths=paths,expected_sha256=hashes,
        project_path=tmp_path/"x.mcscan",output_path=tmp_path/"x.json",
        telemetry_path=tmp_path/"heartbeat.json",api=_api([]))
    telemetry=json.loads((tmp_path/"heartbeat.json").read_text())
    assert telemetry["acceptance"] is False and telemetry["terminal"] is True
    assert len(telemetry["events"])==9 and result["future_context_estimate"]["estimated_bytes"]==1536
    assert result["future_context_estimate"]["allocated"] is False
    elapsed=[row["elapsed_seconds"] for row in telemetry["events"]]
    assert elapsed==sorted(elapsed)
    assert not list(tmp_path.glob("*.partial"))


def test_timeout_publishes_terminal_telemetry_not_acceptance(tmp_path):
    heartbeat=tmp_path/"heartbeat.json"; heartbeat.write_text(json.dumps({"acceptance":False,"events":[
        {"stage":"entry","state":"complete","rss_bytes":10},
        {"stage":"lattice","state":"started","rss_bytes":20}]}))
    mp=_manifest(tmp_path,tmp_path/"out.json",.01)
    fake=lambda *a,**k:(124,"deadline_exhausted",{"job_empty":True},.01)
    code=subject.run_child(["unused"],deadline_seconds=.01,terminal_path=tmp_path/"terminal.json",heartbeat_path=heartbeat,
        manifest_path=mp,stdout_path=tmp_path/"stdout",stderr_path=tmp_path/"stderr",native_execute=fake)
    terminal=json.loads((tmp_path/"terminal.json").read_text())
    assert code==124 and terminal["reason"]=="deadline_exhausted"
    assert terminal["acceptance"] is False and not list(tmp_path.glob("*.partial"))
    assert terminal["last_started_stage"]=="lattice"
    assert terminal["last_completed_stage"]=="entry" and terminal["rss_bytes"]==20


def test_child_failure_publishes_typed_nonaccepting_telemetry(tmp_path):
    heartbeat=tmp_path/"heartbeat.json"; heartbeat.write_text('{"acceptance":false}')
    mp=_manifest(tmp_path,tmp_path/"out.json",2)
    fake=lambda *a,**k:(3,"child_failure",{"job_empty":True},.02)
    code=subject.run_child(["unused"],deadline_seconds=2,terminal_path=tmp_path/"terminal.json",heartbeat_path=heartbeat,
        manifest_path=mp,stdout_path=tmp_path/"stdout",stderr_path=tmp_path/"stderr",native_execute=fake)
    terminal=json.loads((tmp_path/"terminal.json").read_text())
    assert code==3 and terminal["reason"]=="child_failure"
    assert terminal["status"]=="refused" and terminal["acceptance"] is False


def test_public_pipeline_interfaces_and_frozen_graph_defaults_are_unchanged():
    source=PATH.read_text()
    assert "max_shift_cells" not in source and "integer_tolerance" not in source
    assert "min_resultant" not in source and "ambiguous_margin" not in source
    assert "replay_signature_sampling" not in source


def test_real_graph_record_serialization_replays_constructor_and_digest():
    graph=_real_unavailable_graph()
    record=subject.canonical_graph_record(graph)
    assert record["result_sha256"]==graph.result_sha256
    assert record["payload"]["matches"]==[]
    assert record["payload"]["component_solution"] is None
    graph,trace,available=subject.validate_graph_outcome((graph,Record({},"c"*64)))
    assert graph.status=="unavailable" and available is False


def test_upstream_refusal_short_circuits_dependent_stages(tmp_path):
    paths,hashes=_inputs(tmp_path); calls=[]; api=_api(calls)
    class Refused: status="unavailable"; reason="geometry_unavailable"; detail="typed"
    api=subject.StageApi(api.entry,lambda **kwargs:Refused(),api.signatures,api.graph)
    result=subject.execute_graph_stage(png_paths=paths,expected_sha256=hashes,
        project_path=tmp_path/"r.mcscan",output_path=tmp_path/"r.json",
        telemetry_path=tmp_path/"r-heartbeat.json",api=api)
    assert result["execution_complete"] is False and result["failed_stage"]=="lattice"
    assert not any(name in ("signatures","graph") for name,_ in calls)


def test_entry_readiness_refusal_short_circuits_lattice(tmp_path):
    paths,hashes=_inputs(tmp_path); calls=[]; api=_api(calls)
    class BadEntry:
        class Bad: status="unavailable"
        graph=Bad(); pose_graph=Bad(); session=object(); reason="entry_unavailable"
    api=subject.StageApi(lambda *a,**k:BadEntry(),api.lattice,api.signatures,api.graph)
    result=subject.execute_graph_stage(png_paths=paths,expected_sha256=hashes,
        project_path=tmp_path/"e.mcscan",output_path=tmp_path/"e.json",
        telemetry_path=tmp_path/"e-heartbeat.json",api=api)
    assert result["failed_stage"]=="entry" and result["reason"]=="entry_unavailable"
    assert not calls


def test_zero_exit_without_validated_artifact_refuses(tmp_path):
    heartbeat=tmp_path/"h.json"; heartbeat.write_text("{}")
    mp=_manifest(tmp_path,tmp_path/"missing.json",1)
    fake=lambda *a,**k:(0,"",{"job_empty":True},.1)
    code=subject.run_child(["unused"],deadline_seconds=1,terminal_path=tmp_path/"t.json",heartbeat_path=heartbeat,
        manifest_path=mp,stdout_path=tmp_path/"o",stderr_path=tmp_path/"e",native_execute=fake)
    assert code==126 and json.loads((tmp_path/"t.json").read_text())["reason"]=="missing_or_invalid_artifact"


def test_manifest_tamper_and_output_alias_refuse(tmp_path):
    paths,hashes=_inputs(tmp_path)
    common=dict(png_paths=paths,expected_sha256=hashes,project_path=tmp_path/"p.mcscan",
        output_path=tmp_path/"out.json",telemetry_path=tmp_path/"h.json",terminal_path=tmp_path/"t.json",
        manifest_path=tmp_path/"m.json",stdout_path=tmp_path/"stdout",stderr_path=tmp_path/"stderr")
    manifest=subject.create_manifest(**common)
    record=json.loads((tmp_path/"m.json").read_text()); record["limit_bytes"]=1; (tmp_path/"m.json").write_text(json.dumps(record))
    with pytest.raises(ValueError,match="manifest digest"):
        subject.run_child(["unused"],deadline_seconds=1,terminal_path=tmp_path/"term2.json",
            heartbeat_path=tmp_path/"none",manifest_path=tmp_path/"m.json",stdout_path=tmp_path/"x",stderr_path=tmp_path/"y",native_execute=lambda *a,**k:None)
    common["manifest_path"]=tmp_path/"fresh-m.json"; common["output_path"]=paths[0]
    with pytest.raises(ValueError,match="collide"):
        subject.create_manifest(**common)


def test_create_manifest_binds_resolved_matcher_path_and_actual_sha256(tmp_path):
    paths,hashes=_inputs(tmp_path)
    matcher=(ROOT/"src/mc_photogrammetry/reconstruction/plane_graph.py").resolve()
    assert matcher.is_file()
    subject.create_manifest(png_paths=paths,expected_sha256=hashes,project_path=tmp_path/"p.mcscan",
        output_path=tmp_path/"out.json",telemetry_path=tmp_path/"h.json",terminal_path=tmp_path/"t.json",
        manifest_path=tmp_path/"m.json",stdout_path=tmp_path/"stdout",stderr_path=tmp_path/"stderr")
    frozen=json.loads((tmp_path/"m.json").read_text())
    rows=[row for row in frozen["producers"] if Path(row["path"])==matcher]
    assert len(rows)==1
    assert rows[0]["path"]==str(matcher)
    assert rows[0]["sha256"]==hashlib.sha256(matcher.read_bytes()).hexdigest()
    claimed=frozen.pop("manifest_sha256")
    computed=hashlib.sha256(json.dumps(frozen,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    assert claimed==computed


def test_create_manifest_binds_the_alias_resolver_source(tmp_path):
    """The canonical reading index/alias resolver is a frozen producer.

    DS-R3-BV section 5: the v4 alias diagnostic resolves canonical plane indices
    and endpoint aliases through ``plane_graph`` and ``lattice_planes``, so the
    manifest must bind the reading-algorithm source, not only the matcher.
    """
    paths,hashes=_inputs(tmp_path)
    resolver=(ROOT/"src/mc_photogrammetry/reconstruction/lattice_planes.py").resolve()
    assert resolver.is_file()
    subject.create_manifest(png_paths=paths,expected_sha256=hashes,
        project_path=tmp_path/"p.mcscan",output_path=tmp_path/"out.json",
        telemetry_path=tmp_path/"h.json",terminal_path=tmp_path/"t.json",
        manifest_path=tmp_path/"m.json",stdout_path=tmp_path/"stdout",stderr_path=tmp_path/"stderr")
    frozen=json.loads((tmp_path/"m.json").read_text())
    rows=[row for row in frozen["producers"] if Path(row["path"])==resolver]
    assert len(rows)==1
    assert rows[0]["sha256"]==hashlib.sha256(resolver.read_bytes()).hexdigest()
    graph=(ROOT/"src/mc_photogrammetry/pipeline/image_derived_graph.py").resolve()
    attribution=(ROOT/"src/mc_photogrammetry/pipeline/image_derived_graph_attribution.py").resolve()
    bound={Path(row["path"]) for row in frozen["producers"]}
    assert {graph,attribution,resolver} <= bound
    paths_bound={row["path"] for row in frozen["producers"]}
    assert len(paths_bound)==len(frozen["producers"])


def test_direct_api_rejects_existing_and_aliased_outputs(tmp_path):
    paths,hashes=_inputs(tmp_path); occupied=tmp_path/"occupied.json"; occupied.write_text("x")
    with pytest.raises(ValueError,match="fresh"):
        subject.execute_graph_stage(png_paths=paths,expected_sha256=hashes,
            project_path=tmp_path/"p.mcscan",output_path=occupied,
            telemetry_path=tmp_path/"h.json",api=_api([]))
    with pytest.raises(ValueError,match="alias"):
        subject.execute_graph_stage(png_paths=paths,expected_sha256=hashes,
            project_path=tmp_path/"p.mcscan",output_path=paths[0],
            telemetry_path=tmp_path/"h.json",api=_api([]))


def test_typed_stage_refusal_survives_parent_validation(tmp_path):
    output=tmp_path/"refusal.json"; value={"schema":subject.SCHEMA,
        "execution_complete":False,"geometry_available":False,"status":"refused",
        "reason":"geometry_unavailable","project_sha256":""}
    raw=json.dumps(value,sort_keys=True,separators=(",",":")).encode(); value["artifact_sha256"]=hashlib.sha256(raw).hexdigest(); output.write_text(json.dumps(value))
    heartbeat=tmp_path/"h.json"; heartbeat.write_text("{}")
    mp=_manifest(tmp_path,output,1); fake=lambda *a,**k:(0,"",{"job_empty":True},.1)
    code=subject.run_child(["unused"],deadline_seconds=1,terminal_path=tmp_path/"t.json",
        heartbeat_path=heartbeat,manifest_path=mp,stdout_path=tmp_path/"o",
        stderr_path=tmp_path/"e",native_execute=fake)
    terminal=json.loads((tmp_path/"t.json").read_text())
    assert code==0 and terminal["status"]=="refused"
    assert terminal["reason"]=="geometry_unavailable" and terminal["execution_complete"] is False


def test_containment_helper_is_bound_and_cli_has_no_uncontained_path():
    source=PATH.read_text()
    assert "descendant_survived_root_exit" in source and "_cleanup_job" in source
    assert '.tmp-run-axis0-component-preflight-replacement.py' in source
    assert 'parser.error("CLI execution requires --contained")' in source
    assert "subprocess.Popen" not in source


@pytest.mark.parametrize("reason,code,cleanup",[
    ("prelaunch_or_containment_failure",125,{"job_empty":True,"detail":"assignment failed"}),
    ("descendant_survived_root_exit",127,{"job_empty":True}),
    ("child_failure",125,{"job_empty":False,"detail":"cleanup failed"}),
    ("prelaunch_or_containment_failure",125,{"job_empty":True,"detail":"GetExitCodeProcess failed"}),
])
def test_native_adapter_failure_stubs_publish_typed_terminal(tmp_path,reason,code,cleanup):
    heartbeat=tmp_path/"h.json"; heartbeat.write_text("{}")
    mp=_manifest(tmp_path,tmp_path/"out.json",1)
    fake=lambda *a,**k:(code,reason,cleanup,.1)
    result=subject.run_child(["unused"],deadline_seconds=1,terminal_path=tmp_path/"t.json",
        heartbeat_path=heartbeat,manifest_path=mp,stdout_path=tmp_path/"o",
        stderr_path=tmp_path/"e",native_execute=fake)
    terminal=json.loads((tmp_path/"t.json").read_text())
    assert result==code and terminal["status"]=="refused"
    assert terminal["reason"]==reason and terminal["cleanup"]==cleanup


def test_empty_manifest_bindings_refuse_before_native_adapter(tmp_path):
    value={"schema":"mc-photogrammetry-evaluation-graph-stage-manifest-v1",
        "pngs":[],"producers":[],"limit_bytes":8*1024**3,"deadline_seconds":1,
        "output_path":str(tmp_path/"out"),"project_path":str(tmp_path/"project")}
    raw=json.dumps(value,sort_keys=True,separators=(",",":")).encode(); value["manifest_sha256"]=hashlib.sha256(raw).hexdigest()
    path=tmp_path/"m.json"; path.write_text(json.dumps(value)); called=[]
    with pytest.raises(ValueError,match="bindings are empty"):
        subject.run_child(["unused"],deadline_seconds=1,terminal_path=tmp_path/"t",
            heartbeat_path=tmp_path/"h",manifest_path=path,stdout_path=tmp_path/"o",
            stderr_path=tmp_path/"e",native_execute=lambda *a,**k:called.append(1))
    assert called==[]


def test_parent_artifact_joins_inputs_project_graph_trace_and_signatures(tmp_path,monkeypatch):
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    from mc_photogrammetry.input.image import load_image
    from mc_photogrammetry.pipeline.image_derived_session import image_sha256
    from mc_photogrammetry.project import store
    paths,hashes=_inputs(tmp_path)
    manifest={"pngs":[{"path":str(p.resolve()),"sha256":d} for p,d in zip(paths,hashes)],
              "producers":[{"path":str(PATH),"sha256":hashlib.sha256(PATH.read_bytes()).hexdigest()}]}
    decoded=[image_sha256(load_image(p)) for p in paths]
    assert all(file_digest != pixel_digest for file_digest,pixel_digest in zip(hashes,decoded))
    frames=[{"source_label":str(p.resolve()),"image_sha256":d} for p,d in zip(paths,decoded)]
    class Loaded: schema_version=2; source_paths=[str(p.resolve()) for p in paths]; image_derived_session={"session_sha256":"1"*64,"frames":frames}
    monkeypatch.setattr(store,"load_project",lambda path:Loaded())
    payload={"terminal":graph_mod.GRAPH_TERMINAL,"status":"unavailable",
             "reason":"insufficient_axis_coverage","session_sha256":"1"*64,"lattice_sha256":"2"*64,
             "signatures_sha256":"3"*64,"signature_digests":["a"*64],
             "matches":[],"component_solution":None}
    graph_sha=graph_mod._sha(payload)
    trace={"schema":"trace","graph_result_sha256":graph_sha,
           "source_digests":{"session":"1"*64,"lattice":"2"*64,"signatures":"3"*64}}
    trace_sha=hashlib.sha256(json.dumps(trace,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    trace["attribution_sha256"]=trace_sha
    value={"inputs":[{"path":str(p.resolve()),"sha256":d} for p,d in zip(paths,hashes)],
        "graph":{"payload":payload,"result_sha256":graph_sha},"graph_sha256":graph_sha,
        "graph_assembled":False,"solution_available":False,"geometry_available":False,
        "signature_identities":[{"signature_sha256":"a"*64}],
        "trace":trace,"trace_sha256":trace_sha}
    subject._validate_completed_artifact(value,manifest,tmp_path/"project.mcscan")
    Loaded.image_derived_session["session_sha256"]="9"*64
    with pytest.raises(ValueError,match="graph and project session"):
        subject._validate_completed_artifact(value,manifest,tmp_path/"project.mcscan")
    Loaded.image_derived_session["session_sha256"]="1"*64
    contradictory=dict(payload); contradictory["status"]="assembled"
    contradictory_sha=graph_mod._sha(contradictory)
    value["graph"]={"payload":contradictory,"result_sha256":contradictory_sha}
    value["graph_sha256"]=contradictory_sha
    with pytest.raises(ValueError,match="status or terminal flags"):
        subject._validate_completed_artifact(value,manifest,tmp_path/"project.mcscan")
    value["graph"]={"payload":payload,"result_sha256":graph_sha}; value["graph_sha256"]=graph_sha
    value["graph_sha256"]="0"*64
    with pytest.raises(ValueError,match="graph digest"):
        subject._validate_completed_artifact(value,manifest,tmp_path/"project.mcscan")


def test_parent_recomputes_decoded_pixel_digest_separately_from_png_hash(tmp_path,monkeypatch):
    from mc_photogrammetry.input.image import load_image
    from mc_photogrammetry.pipeline.image_derived_session import image_sha256
    from mc_photogrammetry.project import store
    paths,hashes=_inputs(tmp_path)
    decoded=[image_sha256(load_image(path)) for path in paths]
    assert hashes[0] != decoded[0]  # Real encoded-file and framed-pixel domains.
    manifest={"pngs":[{"path":str(p.resolve()),"sha256":d} for p,d in zip(paths,hashes)],
              "producers":[{"path":str(PATH),"sha256":hashlib.sha256(PATH.read_bytes()).hexdigest()}]}
    frames=[{"source_label":str(p.resolve()),"image_sha256":d} for p,d in zip(paths,decoded)]
    class Loaded: schema_version=2; source_paths=[str(p.resolve()) for p in paths]; image_derived_session={"session_sha256":"1"*64,"frames":frames}
    monkeypatch.setattr(store,"load_project",lambda path:Loaded())
    # Stop after the session join: this proves unequal file/pixel hashes pass
    # their correct domains before later graph validation runs.
    with pytest.raises(ValueError,match="inner or outer graph digest"):
        subject._validate_completed_artifact({"inputs":manifest["pngs"],"graph":{}},manifest,tmp_path/"p.mcscan")
    frames[0]["image_sha256"]=hashes[0]
    with pytest.raises(ValueError,match="decoded image binding"):
        subject._validate_completed_artifact({"inputs":manifest["pngs"],"graph":{}},manifest,tmp_path/"p.mcscan")


def test_real_schema2_checkpoint_and_pngs_join_without_hash_domain_confusion():
    from mc_photogrammetry.input.image import load_image
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    from mc_photogrammetry.pipeline.image_derived_session import image_sha256
    from mc_photogrammetry.project.store import load_project
    project=ROOT/".tmp-axis0-component-preflight-replacement-20261002.mcscan"
    loaded=load_project(project); paths=[Path(path) for path in loaded.source_paths]
    assert loaded.schema_version==2 and len(paths)==4 and all(path.is_file() for path in paths)
    file_hashes=[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    pixel_hashes=[image_sha256(load_image(path)) for path in paths]
    assert file_hashes[0]!=pixel_hashes[0]
    assert [row["image_sha256"] for row in loaded.image_derived_session["frames"]]==pixel_hashes
    session_sha=loaded.image_derived_session["session_sha256"]
    payload={"terminal":graph_mod.GRAPH_TERMINAL,"status":"unavailable","reason":"fixture_control",
        "session_sha256":session_sha,"lattice_sha256":"2"*64,"signatures_sha256":"3"*64,
        "signature_digests":["a"*64],"matches":[],"component_solution":None}
    graph_sha=graph_mod._sha(payload)
    trace={"schema":"trace","graph_result_sha256":graph_sha,
        "source_digests":{"session":session_sha,"lattice":"2"*64,"signatures":"3"*64}}
    trace_sha=hashlib.sha256(json.dumps(trace,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    trace["attribution_sha256"]=trace_sha
    rows=[{"path":str(path),"sha256":digest} for path,digest in zip(paths,file_hashes)]
    value={"inputs":rows,"graph":{"payload":payload,"result_sha256":graph_sha},
        "graph_sha256":graph_sha,"graph_assembled":False,"solution_available":False,
        "geometry_available":False,"signature_identities":[{"signature_sha256":"a"*64}],
        "trace":trace,"trace_sha256":trace_sha}
    subject._validate_completed_artifact(value,{"pngs":rows,"producers":[{"bound":True}]},project)


def test_native_inner_exit_code_query_failure_cleans_job(tmp_path,monkeypatch):
    import ctypes
    real=subject._load_containment(ROOT); cleanup=[]
    class Kernel:
        def CreateJobObjectW(self,*a): return 11
        def SetInformationJobObject(self,*a): return 1
        def CreateProcessW(self,*args):
            info=args[-1]._obj; info.hProcess=22; info.hThread=33; return 1
        def AssignProcessToJobObject(self,*a): return 1
        def ResumeThread(self,*a): return 1
        def WaitForSingleObject(self,*a): return real.WAIT_OBJECT_0
        def GetExitCodeProcess(self,*a): return 0
        def CloseHandle(self,*a): return 1
    kernel=Kernel()
    real._kernel32=lambda:kernel
    real._cleanup_job=lambda k,j,p,c:cleanup.append((j,p,c)) or {"job_empty":True}
    monkeypatch.setattr(subject,"_load_containment",lambda root:real)
    code,reason,state,_=subject._native_execute_inner([sys.executable,"-c","pass"],
        deadline_seconds=1,root=ROOT,stdout_path=tmp_path/"stdout",stderr_path=tmp_path/"stderr")
    assert (code,reason)==(125,"exit_code_query_failure")
    assert state["job_empty"] is True and cleanup==[(11,22,125)]


@pytest.mark.parametrize("position",[0,1])
def test_typed_refusal_in_either_graph_tuple_position_short_circuits(tmp_path,position):
    from mc_photogrammetry.reconstruction import plane_placement
    paths,hashes=_inputs(tmp_path); calls=[]; api=_api(calls)
    refusal=plane_placement._make_refusal("provenance_mismatch",{"field":"graph"})
    pair=[_real_unavailable_graph(),Record({},"c"*64)]; pair[position]=refusal
    api=subject.StageApi(api.entry,api.lattice,api.signatures,lambda **kwargs:tuple(pair))
    result=subject.execute_graph_stage(png_paths=paths,expected_sha256=hashes,
        project_path=tmp_path/"g.mcscan",output_path=tmp_path/"g.json",
        telemetry_path=tmp_path/"g-heartbeat.json",api=api)
    assert result["execution_complete"] is False and result["geometry_available"] is False
    assert result["failed_stage"]=="graph" and result["reason"]=="provenance_mismatch"
    assert result["refusal"]==refusal.to_record()
    assert result["detail"]==refusal.detail


def test_parent_terminal_preserves_exact_typed_refusal_record(tmp_path):
    from mc_photogrammetry.reconstruction import plane_placement
    refusal=plane_placement._make_refusal("provenance_mismatch",{"field":"graph","index":2})
    output=tmp_path/"out.json"; mp=_manifest(tmp_path,output,1)
    frozen=json.loads(mp.read_text())
    value={"schema":subject.SCHEMA,"execution_complete":False,"geometry_available":False,
        "status":"refused","reason":refusal.reason,"failed_stage":"graph",
        "detail":refusal.detail,"refusal":refusal.to_record(),"inputs":frozen["pngs"],
        "project_sha256":""}
    raw=json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    value["artifact_sha256"]=hashlib.sha256(raw).hexdigest(); output.write_text(json.dumps(value))
    heartbeat=tmp_path/"h.json"; heartbeat.write_text("{}")
    code=subject.run_child(["unused"],deadline_seconds=1,terminal_path=tmp_path/"terminal.json",
        heartbeat_path=heartbeat,manifest_path=mp,stdout_path=tmp_path/"o",stderr_path=tmp_path/"e",
        native_execute=lambda *a,**k:(0,"",{"job_empty":True},.1))
    terminal=json.loads((tmp_path/"terminal.json").read_text())
    assert code==0 and terminal["refusal"]==refusal.to_record()
    assert terminal["reason"]==refusal.reason and terminal["execution_complete"] is False


def test_three_frame_fixture_is_accepted_and_four_frame_contract_is_preserved(tmp_path):
    three_paths,three_hashes=_inputs(tmp_path,3)
    three=subject.execute_graph_stage(png_paths=three_paths,expected_sha256=three_hashes,
        project_path=tmp_path/"three.mcscan",output_path=tmp_path/"three.json",
        telemetry_path=tmp_path/"three-heartbeat.json",api=_api([]))
    assert three["execution_complete"] is True and three["status"]=="complete"
    assert [row["frame_id"] for row in three["inputs"]]==[
        "image-0000","image-0001","image-0002"]
    assert len(three["inputs"])==3 and three["solver_failure_observation"] is None
    four_paths,four_hashes=_inputs(tmp_path,4)
    four=subject.execute_graph_stage(png_paths=four_paths,expected_sha256=four_hashes,
        project_path=tmp_path/"four.mcscan",output_path=tmp_path/"four.json",
        telemetry_path=tmp_path/"four-heartbeat.json",api=_api([]))
    assert four["execution_complete"] is True and len(four["inputs"])==4
    # The four-frame record is the three-frame record plus one more bound input:
    # every other contract key and value is unchanged.
    assert set(four)==set(three)
    assert four["graph"]==three["graph"] and four["trace"]==three["trace"]
    assert four["solver_failure_observation"]==three["solver_failure_observation"] is None
    assert {row["frame_id"] for row in four["inputs"]}=={
        "image-0000","image-0001","image-0002","image-0003"}


@pytest.mark.parametrize("count",(2,5))
def test_unpinned_frame_counts_refuse_before_any_stage(tmp_path,count):
    paths,hashes=_inputs(tmp_path,count); calls=[]
    with pytest.raises(ValueError,match="three or four"):
        subject.execute_graph_stage(png_paths=paths,expected_sha256=hashes,
            project_path=tmp_path/"x.mcscan",output_path=tmp_path/"x.json",
            telemetry_path=tmp_path/"x-heartbeat.json",api=_api(calls))
    assert calls==[] and not list(tmp_path.glob("*.partial"))


def test_three_frame_pin_mismatch_and_output_collision_fail_closed(tmp_path):
    paths,hashes=_inputs(tmp_path,3)
    tampered=list(hashes); tampered[0]="0"*64; calls=[]
    with pytest.raises(ValueError,match="provenance mismatch"):
        subject.execute_graph_stage(png_paths=paths,expected_sha256=tampered,
            project_path=tmp_path/"pin.mcscan",output_path=tmp_path/"pin.json",
            telemetry_path=tmp_path/"pin-heartbeat.json",api=_api(calls))
    assert calls==[]
    with pytest.raises(ValueError,match="alias"):
        subject.execute_graph_stage(png_paths=paths,expected_sha256=hashes,
            project_path=tmp_path/"alias.mcscan",output_path=paths[0],
            telemetry_path=tmp_path/"alias-heartbeat.json",api=_api([]))
    occupied=tmp_path/"occupied.json"; occupied.write_text("x")
    with pytest.raises(ValueError,match="fresh"):
        subject.execute_graph_stage(png_paths=paths,expected_sha256=hashes,
            project_path=tmp_path/"occupied.mcscan",output_path=occupied,
            telemetry_path=tmp_path/"occupied-heartbeat.json",api=_api([]))
    common=dict(png_paths=paths,expected_sha256=hashes,project_path=tmp_path/"m.mcscan",
        output_path=tmp_path/"m.json",telemetry_path=tmp_path/"mh.json",terminal_path=tmp_path/"mt.json",
        manifest_path=tmp_path/"mm.json",stdout_path=tmp_path/"mo",stderr_path=tmp_path/"me")
    frozen=subject.create_manifest(**common)
    assert len(frozen["pngs"])==3
    common["output_path"]=paths[1]; common["manifest_path"]=tmp_path/"mm2.json"
    with pytest.raises(ValueError,match="collide"):
        subject.create_manifest(**common)


def test_observer_retains_raw_snapshot_when_projection_refuses_partial_axes(tmp_path):
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    from mc_photogrammetry.pipeline import image_derived_graph_attribution as attribution_mod
    paths,hashes=_inputs(tmp_path,3)
    snapshot=graph_mod.SolverFailureSnapshot(
        "plane_integer_residual",("image-0000","image-0001","image-0002"),(0,0,0),
        ((0,0.5),(1,0.95)),((0,0.1),(1,0.12)),(0,0,0),(2,3,4))
    projected=attribution_mod.solver_failure_diagnostic(snapshot,0.90,0.15,0.02)
    assert projected.recorded is False
    assert projected.field=="plane_integer_residual"
    seen=[]
    base=_api([])
    def graph(**kwargs):
        seen.append(kwargs["solver_failure_observer"])
        kwargs["solver_failure_observer"](snapshot)
        return _real_unavailable_graph(),Record(
            {"solver_failure":projected.to_record(),"graph_result_sha256":"c"*64},"c"*64)
    api=subject.StageApi(base.entry,base.lattice,base.signatures,graph)
    result=subject.execute_graph_stage(png_paths=paths,expected_sha256=hashes,
        project_path=tmp_path/"obs.mcscan",output_path=tmp_path/"obs.json",
        telemetry_path=tmp_path/"obs-heartbeat.json",api=api)
    assert callable(seen[0])
    observation=result["solver_failure_observation"]
    assert observation["schema"]==subject.OBSERVATION_SCHEMA
    assert observation["frame_count"]==3
    assert observation["raw"]["schema"]==attribution_mod.SOLVER_FAILURE_OBSERVATION_SCHEMA
    assert observation["raw"]["reason"]=="plane_integer_residual"
    assert observation["raw"]["axis_resultants"]==[
        {"axis":0,"value":0.5},{"axis":1,"value":0.95}]
    assert observation["raw"]["axis_residuals"]==[
        {"axis":0,"value":0.1},{"axis":1,"value":0.12}]
    assert observation["raw"]["best_alternative_axis_observations"]==[2,3,4]
    assert observation["projected"]==projected.to_record()
    assert observation["projected"]["refused"] is True
    assert observation["binding"]["input_sha256"]==hashes
    body={key:value for key,value in observation.items() if key!="observation_sha256"}
    assert observation["observation_sha256"]==hashlib.sha256(json.dumps(
        body,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def test_runner_and_seam_reference_no_truth_or_manual_metadata():
    seam=ROOT/"src/mc_photogrammetry/pipeline/image_derived_graph_attribution.py"
    forbidden=("truth","litematic","camera","scene","reference","bounds",
               "fixture-pins","evaluator-fixture","mask","manual")
    for path in (PATH,seam):
        source=path.read_text(encoding="utf-8").lower()
        assert [token for token in forbidden if token in source]==[]


def test_parent_validates_three_frame_observation_binding(tmp_path,monkeypatch):
    from mc_photogrammetry.input.image import load_image
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    from mc_photogrammetry.pipeline import image_derived_graph_attribution as attribution_mod
    from mc_photogrammetry.pipeline.image_derived_session import image_sha256
    from mc_photogrammetry.project import store
    paths,hashes=_inputs(tmp_path,3)
    manifest={"pngs":[{"path":str(p.resolve()),"sha256":d} for p,d in zip(paths,hashes)],
              "producers":[{"path":str(PATH),"sha256":hashlib.sha256(PATH.read_bytes()).hexdigest()}]}
    decoded=[image_sha256(load_image(path)) for path in paths]
    frames=[{"source_label":str(path.resolve()),"image_sha256":digest}
            for path,digest in zip(paths,decoded)]
    class Loaded:
        schema_version=2
        source_paths=[str(path.resolve()) for path in paths]
        image_derived_session={"session_sha256":"1"*64,"frames":frames}
    monkeypatch.setattr(store,"load_project",lambda path:Loaded())
    snapshot=graph_mod.SolverFailureSnapshot(
        "plane_integer_residual",("a","b","c"),(0,0,0),
        ((0,0.5),(1,0.95)),((0,0.1),(1,0.12)),(0,0,0),(2,3,4))
    projected=attribution_mod.solver_failure_diagnostic(
        snapshot,0.90,0.15,0.02).to_record()
    payload={"terminal":graph_mod.GRAPH_TERMINAL,"status":"unavailable",
        "reason":"plane_integer_residual","session_sha256":"1"*64,
        "lattice_sha256":"2"*64,"signatures_sha256":"3"*64,
        "signature_digests":["a"*64],"matches":[],"component_solution":None}
    graph_sha=graph_mod._sha(payload)
    trace={"schema":"trace","graph_result_sha256":graph_sha,
        "source_digests":{"session":"1"*64,"lattice":"2"*64,"signatures":"3"*64},
        "solver_failure":projected}
    trace_sha=hashlib.sha256(json.dumps(
        trace,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    trace["attribution_sha256"]=trace_sha
    observation={"schema":subject.OBSERVATION_SCHEMA,"frame_count":3,
        "raw":attribution_mod.solver_failure_snapshot_record(snapshot),
        "projected":projected,
        "binding":{"input_sha256":hashes,"entry_sha256":"e"*64,
            "lattice_sha256":"2"*64,"signatures_sha256":"3"*64,"graph_sha256":graph_sha}}
    body=dict(observation)
    observation["observation_sha256"]=hashlib.sha256(json.dumps(
        body,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    value={"inputs":[{"path":str(path.resolve()),"sha256":digest}
                     for path,digest in zip(paths,hashes)],
        "graph":{"payload":payload,"result_sha256":graph_sha},"graph_sha256":graph_sha,
        "graph_assembled":False,"solution_available":False,"geometry_available":False,
        "signature_identities":[{"signature_sha256":"a"*64}],
        "trace":trace,"trace_sha256":trace_sha,"entry_sha256":"e"*64,
        "solver_failure_observation":observation}
    subject._validate_completed_artifact(value,manifest,tmp_path/"project.mcscan")
    tampered=dict(observation)
    tampered["binding"]=dict(observation["binding"]); tampered["binding"]["lattice_sha256"]="9"*64
    tampered_body={key:item for key,item in tampered.items() if key!="observation_sha256"}
    tampered["observation_sha256"]=hashlib.sha256(json.dumps(
        tampered_body,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    value["solver_failure_observation"]=tampered
    with pytest.raises(ValueError,match="observation binding"):
        subject._validate_completed_artifact(value,manifest,tmp_path/"project.mcscan")


# ---------------------------------------------------------------------------
# DS-IMG-SAMEAXIS-PREFILTER-RUNNER-WIRING-R10
#
# The runner attaches the optional pre-filter candidate sink on its single graph
# call, serializes the one record it produced under the explicit top-level
# ``same_axis_candidates`` key (omitted when the solver never failed), replays it
# through the seam's fail-closed validator, and cross-binds its two shared
# digests to the projected active-edge alias record.
# ---------------------------------------------------------------------------

def _same_axis_candidate_record(input_match_identity_sha256=None):
    """One minimal complete V2 all-axis capture record, built only from seams.

    Two local-axis-1 readings in two frames whose assignment places both frames at
    quadrant 0, so local axis 1 maps to world axis 1 and one kept same-axis
    candidate at relative quadrant 0 satisfies the strict v2 replay: it re-derives
    the delta, the kept disposition, the retained-index subset, the ordered input
    identity digest and the shared assignment digest. ``input_match_identity_sha256``
    defaults to the digest the record's own identity actually replays to, because a
    v2 complete record binds the full ordered post-filter input identity in its
    ``input_identities`` list; a caller may still override it to exercise the
    runner's cross-binding.
    """
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    frames=("image-0000","image-0001"); assignments=(0,0)
    readings=([0,frames[0],0,1,-1,[[1.0,3.0,1.0]]],
              [1,frames[1],0,1,1,[[1.0,4.0,1.0]]])
    identity=[frames[0],0,1,frames[1],0,1,0,1.0,1.0]
    if input_match_identity_sha256 is None:
        input_match_identity_sha256=graph_mod._alias_domain_digest(
            graph_mod.SOLVER_INPUT_BINDINGS_INPUT_MATCH_IDENTITY_DOMAIN,
            graph_mod.SOLVER_INPUT_BINDINGS_INPUT_MATCH_IDENTITY_SCHEMA,
            [identity])
    # delta = side_left*distance_left - sigma*side_right*distance_right, sigma=+1
    # for every local-axis-1 endpoint.
    delta=-1.0*3.0-1.0*1.0*4.0
    candidate=[0,0,0,0,0,1,0,0,0.9,5.0,False,1.0,0.1,64,0,0,delta,"kept",""]
    return graph_mod.SameAxisCandidateCapture(
        schema=graph_mod.SAME_AXIS_CANDIDATES_SCHEMA_V2,recorded=True,
        basis=graph_mod.SAME_AXIS_CANDIDATES_BASIS_V2,refusal_reason=None,
        failed_assignment_sha256=graph_mod._failed_assignment_digest(
            frames,dict(zip(frames,assignments))),
        input_match_identity_sha256=input_match_identity_sha256,
        input_match_count=1,group_count=1,row_count=1,frames=frames,
        assignments=assignments,readings=tuple(readings),
        groups=([0,frames[0],frames[1],1,0,[0],[identity]],),
        candidates=(candidate,),
        axes=graph_mod.SAME_AXIS_CANDIDATES_AXES_ALL,
        input_identities=(tuple(identity),))


def _same_axis_candidate_payload(input_match_identity_sha256=None):
    return _same_axis_candidate_record(input_match_identity_sha256).to_record()


def _resign_same_axis_candidate(record):
    """Reseal one capture payload exactly as the seam's validator recomputes it."""
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    body={key:value for key,value in record.items() if key!="record_sha256"}
    domain=(graph_mod.SAME_AXIS_CANDIDATES_DIGEST_DOMAIN_V2
            if record["schema"]==graph_mod.SAME_AXIS_CANDIDATES_SCHEMA_V2
            else graph_mod.SAME_AXIS_CANDIDATES_DIGEST_DOMAIN)
    record["record_sha256"]=hashlib.sha256(json.dumps(
        {"domain":domain,
         "schema":record["schema"],"payload":body},
        sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")).hexdigest()
    return record


def _same_axis_alias_binding(record):
    """The two shared digests as the projected active-edge alias record retains them."""
    return {"schema":"mc-photogrammetry-failed-active-edge-aliases-v1",
        "status":"complete",
        "failed_assignment_sha256":record["failed_assignment_sha256"],
        "input_match_identity_sha256":record["input_match_identity_sha256"]}


def _solver_failure_snapshot():
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    return graph_mod.SolverFailureSnapshot(
        "plane_integer_residual",("image-0000","image-0001","image-0002"),(0,0,0),
        ((0,0.5),(1,0.95)),((0,0.1),(1,0.12)),(0,0,0),(2,3,4))


def _completed_artifact_value(paths,hashes,*,record=None,aliases=None):
    """The reviewed completed-artifact scaffold with an optional capture record."""
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    from mc_photogrammetry.pipeline import image_derived_graph_attribution as attribution_mod
    snapshot=_solver_failure_snapshot()
    projected=attribution_mod.solver_failure_diagnostic(
        snapshot,0.90,0.15,0.02).to_record()
    if aliases is not None:
        projected["active_edge_aliases"]=aliases
    payload={"terminal":graph_mod.GRAPH_TERMINAL,"status":"unavailable",
        "reason":"plane_integer_residual","session_sha256":"1"*64,
        "lattice_sha256":"2"*64,"signatures_sha256":"3"*64,
        "signature_digests":["a"*64],"matches":[],"component_solution":None}
    graph_sha=graph_mod._sha(payload)
    trace={"schema":"trace","graph_result_sha256":graph_sha,
        "source_digests":{"session":"1"*64,"lattice":"2"*64,"signatures":"3"*64},
        "solver_failure":projected}
    trace_sha=hashlib.sha256(json.dumps(
        trace,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    trace["attribution_sha256"]=trace_sha
    observation={"schema":subject.OBSERVATION_SCHEMA,"frame_count":3,
        "raw":attribution_mod.solver_failure_snapshot_record(snapshot),
        "projected":projected,
        "binding":{"input_sha256":hashes,"entry_sha256":"e"*64,
            "lattice_sha256":"2"*64,"signatures_sha256":"3"*64,"graph_sha256":graph_sha}}
    body=dict(observation)
    observation["observation_sha256"]=hashlib.sha256(json.dumps(
        body,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    manifest={"pngs":[{"path":str(p.resolve()),"sha256":d}
                      for p,d in zip(paths,hashes)],
              "producers":[{"path":str(PATH),
                            "sha256":hashlib.sha256(PATH.read_bytes()).hexdigest()}]}
    value={"inputs":[{"path":str(p.resolve()),"sha256":d}
                     for p,d in zip(paths,hashes)],
        "graph":{"payload":payload,"result_sha256":graph_sha},"graph_sha256":graph_sha,
        "graph_assembled":False,"solution_available":False,"geometry_available":False,
        "signature_identities":[{"signature_sha256":"a"*64}],
        "trace":trace,"trace_sha256":trace_sha,"entry_sha256":"e"*64,
        "solver_failure_observation":observation}
    if record is not None:
        value["same_axis_candidates"]=record
    return value,manifest


def test_runner_wires_and_serializes_present_same_axis_candidate_record(tmp_path):
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    paths,hashes=_inputs(tmp_path,3)
    record=_same_axis_candidate_record()
    payload=record.to_record()
    projected={"reason":"plane_integer_residual",
        "active_edge_aliases":_same_axis_alias_binding(payload)}
    seen={}
    base=_api([])

    def graph(**kwargs):
        seen.update(kwargs)
        kwargs["solver_failure_observer"](_solver_failure_snapshot())
        kwargs["same_axis_candidate_sink"](record)
        return _real_unavailable_graph(),Record(
            {"solver_failure":projected,"graph_result_sha256":"c"*64},"c"*64)

    api=subject.StageApi(base.entry,base.lattice,base.signatures,graph)
    result=subject.execute_graph_stage(png_paths=paths,expected_sha256=hashes,
        project_path=tmp_path/"cap.mcscan",output_path=tmp_path/"cap.json",
        telemetry_path=tmp_path/"cap-heartbeat.json",api=api)
    assert callable(seen["same_axis_candidate_sink"])
    assert seen["matcher_witness_sink"] is None
    # DS-IMG-SAMEAXIS-ALLAXIS-CAPTURE-V2: the runner requests the explicit
    # all-world-axis aggregate from the seam, never a partial axis subset.
    assert seen["same_axis_capture_axes"]==subject.ALL_AXIS_CAPTURE_AXES==(0,1,2)
    assert result["solver_failure_observation"]["projected"]==projected
    assert result["same_axis_candidates"]==payload
    assert payload["schema"]==graph_mod.SAME_AXIS_CANDIDATES_SCHEMA_V2
    assert payload["axes"]==[0,1,2]
    graph_mod.validate_same_axis_candidates(result["same_axis_candidates"])


def test_runner_omits_same_axis_record_without_a_solver_failure(tmp_path):
    paths,hashes=_inputs(tmp_path,3)
    default=subject.execute_graph_stage(png_paths=paths,expected_sha256=hashes,
        project_path=tmp_path/"d.mcscan",output_path=tmp_path/"d.json",
        telemetry_path=tmp_path/"d-heartbeat.json",api=_api([]))
    assert default["solver_failure_observation"] is None
    assert "same_axis_candidates" not in default
    base=_api([])

    def graph(**kwargs):
        kwargs["solver_failure_observer"](_solver_failure_snapshot())
        return _real_unavailable_graph(),Record(
            {"solver_failure":{"reason":"plane_integer_residual"}},"c"*64)

    api=subject.StageApi(base.entry,base.lattice,base.signatures,graph)
    observed=subject.execute_graph_stage(png_paths=paths,expected_sha256=hashes,
        project_path=tmp_path/"o.mcscan",output_path=tmp_path/"o.json",
        telemetry_path=tmp_path/"o-heartbeat.json",api=api)
    assert observed["solver_failure_observation"] is not None
    assert "same_axis_candidates" not in observed


def test_runner_propagates_typed_same_axis_refusal(tmp_path):
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    paths,hashes=_inputs(tmp_path,3)
    refusal=graph_mod.failed_same_axis_candidates(None,None,None,None)
    assert refusal.recorded is False
    projected={"reason":"plane_integer_residual"}
    base=_api([])

    def graph(**kwargs):
        kwargs["solver_failure_observer"](_solver_failure_snapshot())
        kwargs["same_axis_candidate_sink"](refusal)
        return _real_unavailable_graph(),Record(
            {"solver_failure":projected},"c"*64)

    api=subject.StageApi(base.entry,base.lattice,base.signatures,graph)
    result=subject.execute_graph_stage(png_paths=paths,expected_sha256=hashes,
        project_path=tmp_path/"r.mcscan",output_path=tmp_path/"r.json",
        telemetry_path=tmp_path/"r-heartbeat.json",api=api)
    record=result["same_axis_candidates"]
    assert record==refusal.to_record()
    assert record["status"]=="refused"
    assert record["reason"]=="invalid_same_axis_candidate_capture"
    assert record["field"]=="observation"
    graph_mod.validate_same_axis_candidates(record)


def test_runner_validates_same_axis_candidate_record_and_shared_bindings(
        tmp_path,monkeypatch):
    from mc_photogrammetry.input.image import load_image
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    from mc_photogrammetry.pipeline.image_derived_session import image_sha256
    from mc_photogrammetry.project import store
    paths,hashes=_inputs(tmp_path,3)
    record=_same_axis_candidate_payload()
    value,manifest=_completed_artifact_value(
        paths,hashes,record=record,aliases=_same_axis_alias_binding(record))
    decoded=[image_sha256(load_image(path)) for path in paths]
    frames=[{"source_label":str(path.resolve()),"image_sha256":digest}
            for path,digest in zip(paths,decoded)]

    class Loaded:
        schema_version=2
        source_paths=[str(path.resolve()) for path in paths]
        image_derived_session={"session_sha256":"1"*64,"frames":frames}

    monkeypatch.setattr(store,"load_project",lambda path:Loaded())
    project=tmp_path/"project.mcscan"
    subject._validate_completed_artifact(value,manifest,project)
    graph_mod.validate_same_axis_candidates(value["same_axis_candidates"])

    # A stale digest fails the seam's own fail-closed replay.
    stale=json.loads(json.dumps(record)); stale["record_sha256"]="0"*64
    value["same_axis_candidates"]=stale
    with pytest.raises(ValueError,match="digest mismatch"):
        subject._validate_completed_artifact(value,manifest,project)

    # A resealed record whose input-match identity no longer names the shared
    # solver-input identity is rejected. For a v2 record the ordered identity is
    # re-derived from the bound groups, so the record's own fail-closed replay
    # catches the tamper before the runner's cross-binding would; either message
    # is an identity failure and the record never reaches acceptance.
    identity=json.loads(json.dumps(record))
    identity["input_match_identity_sha256"]="d"*64
    value["same_axis_candidates"]=_resign_same_axis_candidate(identity)
    with pytest.raises(ValueError,match="identity"):
        subject._validate_completed_artifact(value,manifest,project)

    # A projected alias record that no longer names the same failed assignment
    # is rejected by the same cross-binding.
    value["same_axis_candidates"]=record
    trace=value["trace"]
    trace["solver_failure"]["active_edge_aliases"]["failed_assignment_sha256"]="e"*64
    with pytest.raises(ValueError,match="assignment mismatch"):
        subject._validate_same_axis_candidates_record(value,trace)
    with pytest.raises(ValueError,match="malformed"):
        subject._validate_same_axis_candidates_record(
            {"same_axis_candidates":"not-a-record"},trace)


def test_runner_rejects_complete_same_axis_record_without_complete_alias_binding(
        tmp_path):
    """DS-IMG-SAMEAXIS-R10-BINDING-REPAIR: a complete capture needs a binding.

    Before the repair a ``status=complete`` candidate record passed whenever the
    projected active-edge alias record was absent, not a mapping, or not
    ``status=complete``, because both shared-digest checks were skipped. Every one
    of those shapes must now fail closed, and only after the binding is proven
    complete are the two digests compared. A typed candidate refusal carries no
    digests, so it must still replay on its own.
    """
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    paths,hashes=_inputs(tmp_path,3)
    record=_same_axis_candidate_payload()
    aliases=_same_axis_alias_binding(record)

    # Control: complete record plus complete alias binding is accepted.
    value,_manifest=_completed_artifact_value(
        paths,hashes,record=record,aliases=aliases)
    subject._validate_same_axis_candidates_record(value,value["trace"])

    # (1) Projected solver failure absent, empty, or not a mapping.
    for trace in ({"solver_failure":None},{"solver_failure":"not-a-failure"},{}):
        with pytest.raises(ValueError,match="projected solver failure"):
            subject._validate_same_axis_candidates_record(value,trace)

    # (2) Solver failure present but its alias binding is missing or wrong-shape,
    # including a well-formed typed alias refusal (incomplete, so not a binding).
    failure={"reason":"plane_integer_residual"}
    refused_alias={"schema":"mc-photogrammetry-failed-active-edge-aliases-v1",
        "status":"refused","reason":"budget_exhausted","field":"record_bytes"}
    for raw_aliases in (None,"not-an-alias-record",[],refused_alias,{}):
        trace={"solver_failure":dict(failure,active_edge_aliases=raw_aliases)}
        with pytest.raises(ValueError,match="complete alias binding"):
            subject._validate_same_axis_candidates_record(value,trace)

    # (3) A complete binding is compared, not merely present: either shared digest
    # naming a different value still trips its own mismatch.
    trace={"solver_failure":dict(failure,active_edge_aliases=dict(aliases))}
    trace["solver_failure"]["active_edge_aliases"]["failed_assignment_sha256"]="e"*64
    with pytest.raises(ValueError,match="assignment mismatch"):
        subject._validate_same_axis_candidates_record(value,trace)
    trace={"solver_failure":dict(failure,active_edge_aliases=dict(aliases))}
    trace["solver_failure"]["active_edge_aliases"]["input_match_identity_sha256"]="d"*64
    with pytest.raises(ValueError,match="input identity mismatch"):
        subject._validate_same_axis_candidates_record(value,trace)

    # (4) Typed candidate-refusal replay is preserved and needs no binding.
    refusal=graph_mod.failed_same_axis_candidates(None,None,None,None)
    refusal_value={"same_axis_candidates":refusal.to_record()}
    assert refusal_value["same_axis_candidates"]["status"]=="refused"
    subject._validate_same_axis_candidates_record(refusal_value,{})


def test_runner_accepts_v1_or_v2_capture_and_retains_cross_binding():
    """DS-IMG-SAMEAXIS-ALLAXIS-CAPTURE-V2: both schemas keep the R10 binding.

    The runner's replay is schema-agnostic but fail-closed: the frozen v1
    single-axis record and the v2 all-axis aggregate are both accepted only when
    both shared digests match a complete projected alias record; either schema
    still fails closed on a missing, refused or digest-mismatched alias binding.
    """
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    v2=graph_mod.SameAxisCandidateCapture(
        schema=graph_mod.SAME_AXIS_CANDIDATES_SCHEMA_V2,recorded=False,
        basis=None,refusal_reason="invalid_same_axis_candidate_capture",
        failed_assignment_sha256="",
        input_match_identity_sha256="",input_match_count=0,group_count=0,
        row_count=0,frames=(),assignments=(),readings=(),groups=(),
        candidates=(),field="observation",
        axes=graph_mod.SAME_AXIS_CANDIDATES_AXES_ALL).to_record()
    v1=_same_axis_candidate_record().to_record()
    v1["schema"]=graph_mod.SAME_AXIS_CANDIDATES_SCHEMA
    v1["basis"]=graph_mod.SAME_AXIS_CANDIDATES_BASIS
    v1.pop("axes")
    v1.pop("input_identities")
    v1["axis"]=graph_mod.SAME_AXIS_CANDIDATES_AXIS
    # v1 binds one scalar survivor index and one identity, not parallel lists.
    v1["groups"]=[[group[0],group[1],group[2],group[3],group[4],
                   group[5][0],group[6][0]] for group in v1["groups"]]
    _resign_same_axis_candidate(v1)
    assert v2["status"]=="refused" and v1["status"]=="complete"
    graph_mod.validate_same_axis_candidates(v1)

    frozen_v2=_same_axis_candidate_payload()
    aliases=_same_axis_alias_binding(frozen_v2)
    for record in (frozen_v2,v1):
        value={"same_axis_candidates":record}
        subject._validate_same_axis_candidates_record(
            value,{"solver_failure":{"reason":"plane_integer_residual",
                                     "active_edge_aliases":aliases}})
        for bad in (None,{"status":"refused"},{}):
            trace={"solver_failure":{"reason":"plane_integer_residual",
                                     "active_edge_aliases":bad}}
            with pytest.raises(ValueError,match="complete alias binding"):
                subject._validate_same_axis_candidates_record(value,trace)
    # A typed v2 refusal replays on its own with no binding.
    refusal_value={"same_axis_candidates":v2}
    assert refusal_value["same_axis_candidates"]["status"]=="refused"
    subject._validate_same_axis_candidates_record(refusal_value,{})


def test_runner_cross_binds_realistic_subset_v2_candidate_record():
    """DS-IMG-SAMEAXIS-ALLAXIS-CAPTURE-V2-CORRECTION: the real subset cardinality.

    The 38-input / 9-group / 10-survivor record (28 inputs no captured group
    contains) must replay through the seam and still cross-bind both shared
    digests to a complete projected alias record, while a missing/refused alias
    binding and either mismatched digest still fail closed.
    """
    from test_image_derived_graph_same_axis_candidate_capture import (
        realistic_v2_subset_payload)
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    payload=realistic_v2_subset_payload()
    graph_mod.validate_same_axis_candidates(payload)
    assert payload["input_match_count"]==38
    assert len(payload["input_identities"])==38
    assert payload["group_count"]==9
    aliases=_same_axis_alias_binding(payload)
    value={"same_axis_candidates":payload}

    subject._validate_same_axis_candidates_record(
        value,{"solver_failure":{"reason":"plane_integer_residual",
                                 "active_edge_aliases":aliases}})

    for missing in (None,{"status":"refused"},{}):
        with pytest.raises(ValueError,match="complete alias binding"):
            subject._validate_same_axis_candidates_record(
                value,{"solver_failure":{"reason":"plane_integer_residual",
                                         "active_edge_aliases":missing}})
    bad_assignment={"solver_failure":{"reason":"plane_integer_residual",
        "active_edge_aliases":dict(aliases,failed_assignment_sha256="e"*64)}}
    with pytest.raises(ValueError,match="assignment mismatch"):
        subject._validate_same_axis_candidates_record(value,bad_assignment)
    bad_identity={"solver_failure":{"reason":"plane_integer_residual",
        "active_edge_aliases":dict(aliases,input_match_identity_sha256="d"*64)}}
    with pytest.raises(ValueError,match="input identity mismatch"):
        subject._validate_same_axis_candidates_record(value,bad_identity)


def test_runner_rejects_resealed_omission_of_an_active_v2_group():
    """DS-IMG-SAMEAXIS-ALLAXIS-CAPTURE-V2-COVERAGE: exact active coverage.

    Removing one entire solver-active group (and its candidate rows) from the
    reviewed 38-input record, then re-sealing it under its own recomputed digest,
    must still be refused through the runner seam because the surviving input
    identities independently imply the omitted group. The unmodified record keeps
    its accepted cross-binding.
    """
    from test_image_derived_graph_same_axis_candidate_capture import (
        realistic_v2_subset_payload)
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    payload=realistic_v2_subset_payload()
    graph_mod.validate_same_axis_candidates(payload)
    aliases=_same_axis_alias_binding(payload)
    value={"same_axis_candidates":payload}
    subject._validate_same_axis_candidates_record(
        value,{"solver_failure":{"reason":"plane_integer_residual",
                                 "active_edge_aliases":aliases}})

    omitted=json.loads(json.dumps(payload))
    omitted["groups"]=omitted["groups"][:-1]
    omitted["group_count"]=len(omitted["groups"])
    remaining=sorted(row[0] for row in omitted["groups"])
    omitted["candidates"]=[row for row in omitted["candidates"]
                           if row[0] in remaining]
    omitted["candidates"]=[[remaining.index(row[0])]+row[1:]
                           for row in omitted["candidates"]]
    omitted["row_count"]=len(omitted["candidates"])
    _resign_same_axis_candidate(omitted)
    assert omitted["input_match_count"]==38
    with pytest.raises(ValueError,match="does not cover the active group set"):
        subject._validate_same_axis_candidates_record(
            {"same_axis_candidates":omitted},
            {"solver_failure":{"reason":"plane_integer_residual",
                               "active_edge_aliases":aliases}})


def test_runner_completed_artifact_rejects_complete_record_without_aliases(
        tmp_path,monkeypatch):
    """The repaired binding is enforced on the real completed-artifact path."""
    from mc_photogrammetry.input.image import load_image
    from mc_photogrammetry.pipeline.image_derived_session import image_sha256
    from mc_photogrammetry.project import store
    paths,hashes=_inputs(tmp_path,3)
    record=_same_axis_candidate_payload()
    # aliases=None means the projected failure is correctly signed with no
    # active_edge_aliases key at all, so the earlier observation/join checks pass
    # and execution reaches the candidate-record validator.
    value,manifest=_completed_artifact_value(paths,hashes,record=record)
    decoded=[image_sha256(load_image(path)) for path in paths]
    frames=[{"source_label":str(path.resolve()),"image_sha256":digest}
            for path,digest in zip(paths,decoded)]

    class Loaded:
        schema_version=2
        source_paths=[str(path.resolve()) for path in paths]
        image_derived_session={"session_sha256":"1"*64,"frames":frames}

    monkeypatch.setattr(store,"load_project",lambda path:Loaded())
    project=tmp_path/"project.mcscan"
    assert "active_edge_aliases" not in value["trace"]["solver_failure"]
    with pytest.raises(ValueError,match="complete alias binding"):
        subject._validate_completed_artifact(value,manifest,project)

