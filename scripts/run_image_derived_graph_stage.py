"""Evaluation-only staged graph runner over three or four frozen PNG inputs."""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Callable

from PIL import Image

from mc_photogrammetry.pipeline import image_derived_graph as graph_mod

SCHEMA = "mc-photogrammetry-evaluation-graph-stage-v1"
TELEMETRY_SCHEMA = "mc-photogrammetry-evaluation-stage-telemetry-v1"
OBSERVATION_SCHEMA = "mc-photogrammetry-evaluation-graph-stage-observation-v1"
CONTEXT_ESTIMATE_CAP = 256 * 1024**2
# Exact pinned fixture cardinalities the reviewed harness accepts.  Four frames
# remain the original contract; three frames are the pinned Template 2 fixture.
ALLOWED_FRAME_COUNTS = (3, 4)
# DS-IMG-SAMEAXIS-ALLAXIS-CAPTURE-V2: the runner requests the evaluation-only
# all-world-axis candidate aggregate. The tuple is pinned here from the seam's own
# frozen constant, so the runner and the producer can never drift apart.
ALL_AXIS_CAPTURE_AXES = graph_mod.SAME_AXIS_CANDIDATES_AXES_ALL
assert ALL_AXIS_CAPTURE_AXES == (0, 1, 2)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _rss() -> int | None:
    try:
        import ctypes
        from ctypes import wintypes
        class PMC(ctypes.Structure):
            _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD),
                ("PeakWorkingSetSize",ctypes.c_size_t),("WorkingSetSize",ctypes.c_size_t),
                ("rest",ctypes.c_size_t*6)]
        kernel=ctypes.WinDLL("kernel32",use_last_error=True); psapi=ctypes.WinDLL("psapi",use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PMC),wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
        row=PMC(); row.cb=ctypes.sizeof(row)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(row),row.cb): return None
        value=int(row.WorkingSetSize); return value if value > 0 else None
    except Exception:
        return None


@dataclass(frozen=True)
class StageApi:
    entry: Callable
    lattice: Callable
    signatures: Callable
    graph: Callable


class StageFailure(RuntimeError):
    def __init__(self, stage, reason, detail="", refusal_record=None):
        super().__init__(detail or reason); self.stage=stage; self.reason=reason; self.detail=detail
        self.refusal_record=refusal_record


def _placement_refusal_record(value):
    """Replay and retain the exact typed refusal record and its digests."""
    from mc_photogrammetry.reconstruction import plane_placement as placement
    if not isinstance(value,placement.PlacementRefusal): return None
    record=value.to_record()
    work_record=record["work"]
    work=None if work_record is None else placement.PlacementWork(**work_record)
    replay=placement.PlacementRefusal(record["reason"],record["detail"],
        record["detail_sha256"],work,record["refusal_sha256"])
    if replay.to_record()!=record: raise ValueError("placement refusal replay mismatch")
    return record


def _default_api() -> StageApi:
    from mc_photogrammetry.pipeline.image_derived_entry import build_image_derived_entry
    from mc_photogrammetry.pipeline.image_derived_lattice_measure import measure_image_derived_lattice
    from mc_photogrammetry.pipeline.image_derived_signatures import derive_image_derived_signatures
    from mc_photogrammetry.pipeline.image_derived_graph_attribution import derive_image_derived_graph_attribution
    return StageApi(build_image_derived_entry, measure_image_derived_lattice,
                    derive_image_derived_signatures, derive_image_derived_graph_attribution)


def canonical_graph_record(graph) -> dict:
    """Serialize and replay the exact v2 graph digest contract."""
    import dataclasses
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    if not isinstance(graph, graph_mod.ImageDerivedGraphResult):
        return {"type":type(graph).__name__,"reason":getattr(graph,"reason","")}
    replay=graph_mod.ImageDerivedGraphResult(**{
        field.name:getattr(graph,field.name) for field in dataclasses.fields(graph)})
    payload=graph_mod._payload(replay)
    computed=graph_mod._sha(payload)
    if computed!=graph.result_sha256 or replay.result_sha256!=graph.result_sha256:
        raise ValueError("graph canonical digest replay mismatch")
    return {"payload":payload,"result_sha256":computed}


def validate_graph_outcome(value):
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    if not isinstance(value,tuple) or len(value)!=2:
        raise StageFailure("graph","invalid_result","graph adapter must return graph and trace")
    graph,trace=value
    for member in value:
        refusal=_placement_refusal_record(member)
        if refusal is not None:
            raise StageFailure("graph",refusal["reason"],refusal["detail"],refusal)
    if not isinstance(graph,graph_mod.ImageDerivedGraphResult):
        raise StageFailure("graph","invalid_result","first graph tuple member is foreign")
    # `terminal` names the typed unavailable terminal; availability is the
    # explicit status/solution invariant enforced by ImageDerivedGraphResult.
    available=(graph.status=="assembled" and graph.component_solution is not None)
    if available!=(graph.status=="assembled"):
        raise StageFailure("graph","provenance_mismatch","graph status/solution mismatch")
    return graph,trace,available


def _solver_failure_observation(observations, identities, entry, lattice, signatures,
                                graph, trace_record):
    """Bind the exact raw failure snapshot to every stage digest.

    The observer callback captures the actual in-memory
    ``SolverFailureSnapshot`` produced by the same graph derivation.  This
    serializes it through the seam's own unprojected record and joins it to the
    projected ``solver_failure`` trace record and to the
    input/entry/lattice/signature/graph digests, so a partial-axis projection
    refusal cannot hide the raw per-axis metrics.  It is ``None`` whenever the
    solver never abstained.
    """
    if not observations:
        return None
    from mc_photogrammetry.pipeline.image_derived_graph_attribution import (
        solver_failure_snapshot_record)
    projected=None if trace_record is None else trace_record.get("solver_failure")
    observation={"schema":OBSERVATION_SCHEMA,"frame_count":len(identities),
        "raw":solver_failure_snapshot_record(observations[0]),
        "projected":projected,
        "binding":{"input_sha256":[row["sha256"] for row in identities],
            "entry_sha256":getattr(entry,"entry_sha256",""),
            "lattice_sha256":getattr(lattice,"result_sha256",""),
            "signatures_sha256":getattr(signatures,"result_sha256",""),
            "graph_sha256":getattr(graph,"result_sha256","")}}
    # DS-IMG-GEOM-PHASE-CAPTURE: mirror the optional per-face phase view that the
    # same failure snapshot already carries.  It is present only when the graph
    # seam produced a phase-capture record, and it is the record's own digest-bound
    # serialization, never a re-derivation from the projected diagnostic.
    phase_capture=getattr(observations[0],"phase_capture",None)
    if phase_capture is not None:
        observation["phase_capture"]=phase_capture.to_record()
    # DS-R7: mirror the optional complete solver-input binding record carried by
    # the same failure snapshot. Like the phase capture it is present only when
    # the graph seam produced it, and it is the record's own digest-bound
    # serialization, never a re-derivation from the projected diagnostic.
    solver_input_bindings=getattr(observations[0],"solver_input_bindings",None)
    if solver_input_bindings is not None:
        observation["solver_input_bindings"]=solver_input_bindings.to_record()
    observation["observation_sha256"]=hashlib.sha256(json.dumps(
        observation,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return observation


def execute_graph_stage(*, png_paths, expected_sha256, project_path, output_path,
                        telemetry_path, api: StageApi | None = None, clock=time.monotonic,
                        _period_support_observer=None):
    """Default call-through; the private sidecar is evaluation-only and inert off."""
    kwargs = dict(png_paths=png_paths, expected_sha256=expected_sha256,
                  project_path=project_path, output_path=output_path,
                  telemetry_path=telemetry_path, api=api, clock=clock)
    if _period_support_observer is None:
        return _execute_graph_stage(**kwargs)
    from mc_photogrammetry.reconstruction import lattice as support_observation
    observer = _period_support_observer
    with support_observation._period_support_scope(observer,
            run_id=observer.run_id, signature_targets=observer.signature_targets,
            match_targets=observer.match_targets) as capture:
        try:
            return _execute_graph_stage(**kwargs)
        finally:
            try:
                observer.finish_capture(capture.incomplete)
            except Exception as error:
                capture.fail(getattr(error, "reason", "capture_failure"), type(error).__name__)
                try:
                    observer.incomplete = dict(capture.incomplete)
                except Exception:
                    pass


def _execute_graph_stage(*, png_paths, expected_sha256, project_path, output_path,
                        telemetry_path, api: StageApi | None = None, clock=time.monotonic):
    from mc_photogrammetry.reconstruction import lattice as support_observation
    paths=tuple(Path(value).resolve() for value in png_paths); expected=tuple(expected_sha256)
    project_path=Path(project_path).resolve(); output_path=Path(output_path).resolve()
    telemetry_path=Path(telemetry_path).resolve()
    targets=(project_path,output_path,telemetry_path)
    if len(set(targets))!=3 or set(paths)&set(targets):
        raise ValueError("stage outputs collide or alias PNG inputs")
    if any(path.exists() for path in targets): raise ValueError("fresh stage outputs are required")
    started=clock(); events=[]
    def beat(stage, state, detail=""):
        events.append({"stage":stage,"state":state,"elapsed_seconds":clock()-started,
                       "rss_bytes":_rss(),"detail":detail})
        _atomic(Path(telemetry_path), {"schema":TELEMETRY_SCHEMA,"acceptance":False,
            "terminal":False,"events":events})
    if len(paths) not in ALLOWED_FRAME_COUNTS or len(expected)!=len(paths) or len(set(paths))!=len(paths):
        raise ValueError("exactly three or four distinct PNG inputs and hashes are required")
    identities=[]
    for index,(path,digest) in enumerate(zip(paths,expected)):
        if not path.is_file() or path.suffix.lower()!=".png" or _sha(path)!=digest:
            raise ValueError(f"PNG provenance mismatch at index {index}")
        with Image.open(path) as image:
            image.verify(); width,height=image.size
        identities.append({"frame_id":f"image-{index:04d}","path":str(path),
                           "sha256":digest,"width":width,"height":height})
    planned=sum(row["width"]*row["height"]*8 for row in identities)
    if planned>CONTEXT_ESTIMATE_CAP: raise ValueError("future context estimate exceeds cap")
    context_estimate={"schema":"mc-photogrammetry-future-context-estimate-v1",
        "estimated_bytes":planned,"comparison_cap_bytes":CONTEXT_ESTIMATE_CAP,
        "allocated":False,"reservation":False,
        "accounting":"estimate only: RGB uint8 + future ownership int32 + usable uint8 per source pixel"}
    api=api or _default_api()
    beat("future_context","estimated")
    def call(stage, function, *args, expected_status=None, **kwargs):
        beat(stage,"started")
        try: value=function(*args,**kwargs)
        except Exception as error:
            raise StageFailure(stage,"exception",f"{type(error).__name__}: {error}") from error
        refusal=_placement_refusal_record(value)
        if refusal is not None:
            raise StageFailure(stage,refusal["reason"],refusal["detail"],refusal)
        if expected_status is not None and getattr(value,"status",None)!=expected_status:
            raise StageFailure(stage,getattr(value,"reason","upstream_unavailable"),getattr(value,"detail",""))
        beat(stage,"complete"); return value
    try:
        entry=call("entry",api.entry,paths,project_path=project_path)
        if (getattr(entry.graph,"status",None)!="ok"
                or getattr(entry.pose_graph,"status",None)!="ok"):
            raise StageFailure("entry",getattr(entry,"reason","entry_unavailable"),
                               "entry graph or pose graph is unavailable")
        def observe_entry(_capture):
            for source_frame in entry.session.frames:
                pin = next(row for row in identities if row["frame_id"] == source_frame.frame_id)
                support_observation._period_support_event("frame_source",
                    input_identity=pin, frame=source_frame.frame_id, rgb=source_frame.image,
                    ingress=(None if source_frame.ingress is None else source_frame.ingress.to_record()),
                    decoded_sha256=source_frame.image_sha256,
                    session_sha256=entry.session.session_sha256)
        support_observation._period_support_private(observe_entry)
        lattice=call("lattice",api.lattice,session=entry.session,pose_graph=entry.pose_graph,
                     expected_status="assembled")
        support_observation._period_support_private(lambda _capture:
            support_observation._period_support_event("lattice_final", readings=lattice.readings,
                lattice_sha256=lattice.result_sha256))
        signatures=call("signatures",api.signatures,session=entry.session,lattice_result=lattice,
                        expected_status="assembled")
    except StageFailure as failure:
        project_sha=_sha(project_path) if project_path.is_file() else ""
        payload={"schema":SCHEMA,"execution_complete":False,"geometry_available":False,
            "status":"refused","reason":failure.reason,"failed_stage":failure.stage,
            "detail":failure.detail,"refusal":failure.refusal_record,
            "inputs":identities,"future_context_estimate":context_estimate,
            "project_sha256":project_sha,"telemetry":{"acceptance":False,"events":events}}
        raw=json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        payload["artifact_sha256"]=hashlib.sha256(raw).hexdigest(); _atomic(output_path,payload)
        return payload
    signature_ids=[{"frame_id":row.frame_id,"canonical_index":row.canonical_index,
        "alternative_index":row.alternative_index,"signature_sha256":row.signature_sha256}
        for row in signatures.signatures]
    observations=[]
    # DS-IMG-SAMEAXIS-PREFILTER-RUNNER-WIRING-R10: the optional pre-filter
    # candidate sink is attached on the same graph call as the failure observer.
    # The graph seam only invokes it when the component solver abstains, so it
    # stays empty on every successful derivation and never changes the returned
    # graph result, its digest or the .mcscan project.
    same_axis_candidates=[]
    def observe(snapshot):
        observations.append(snapshot)
    def observe_same_axis(record):
        same_axis_candidates.append(record)
    try:
        graph_value=call("graph",api.graph,session=entry.session,lattice_result=lattice,
            signatures_result=signatures,connection_graph=entry.graph,
            matcher_witness_sink=None,matcher_witness_payload_sink=None,witness_budget=None,
            solver_failure_observer=observe,same_axis_candidate_sink=observe_same_axis,
            same_axis_capture_axes=ALL_AXIS_CAPTURE_AXES)
        graph,trace,graph_assembled=validate_graph_outcome(graph_value)
    except StageFailure as failure:
        project_sha=_sha(project_path) if project_path.is_file() else ""
        payload={"schema":SCHEMA,"execution_complete":False,"geometry_available":False,
            "status":"refused","reason":failure.reason,"failed_stage":failure.stage,
            "detail":failure.detail,"refusal":failure.refusal_record,
            "inputs":identities,"signature_identities":signature_ids,
            "future_context_estimate":context_estimate,"project_sha256":project_sha,
            "telemetry":{"acceptance":False,"events":events}}
        raw=json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        payload["artifact_sha256"]=hashlib.sha256(raw).hexdigest(); _atomic(output_path,payload)
        return payload
    graph_record=canonical_graph_record(graph)
    trace_record=None if trace is None else trace.to_record()
    observation=_solver_failure_observation(
        observations, identities, entry, lattice, signatures, graph, trace_record)
    # DS-IMG-SAMEAXIS-PREFILTER-RUNNER-WIRING-R10 /
    # DS-IMG-SAMEAXIS-ALLAXIS-CAPTURE-V2: serialize the single record the optional
    # pre-filter candidate sink produced, when the solver abstained, under its own
    # explicit top-level key. This runner requests the all-world-axis v2 aggregate
    # (``ALL_AXIS_CAPTURE_AXES``); the key is the record's own digest-bound payload,
    # never a re-derivation, and it is omitted entirely when no solver failure (and
    # therefore no record) occurred. Nothing about the public graph result, its
    # digest, the .mcscan project or the nested SolverFailureSnapshot changes.
    same_axis_record=(None if not same_axis_candidates
                      else same_axis_candidates[0].to_record())
    payload={"schema":SCHEMA,"execution_complete":True,
        "geometry_available":False,"graph_assembled":graph_assembled,
        "solution_available":graph.component_solution is not None,
        "status":"complete","reason":"","inputs":identities,
        "future_context_estimate":context_estimate,"signature_identities":signature_ids,
        "graph":graph_record,"trace":trace_record,"graph_sha256":getattr(graph,"result_sha256",""),
        "trace_sha256":getattr(trace,"attribution_sha256",""),"project_sha256":_sha(project_path),
        "entry_sha256":getattr(entry,"entry_sha256",""),
        "solver_failure_observation":observation,
        "telemetry":{"acceptance":False,"events":events}}
    if same_axis_record is not None:
        payload["same_axis_candidates"]=same_axis_record
    support_observation._period_support_private(lambda _capture:
        support_observation._period_support_event("core_final",
            graph=graph_record, trace=trace_record, solver_failure_observation=observation,
            graph_sha256=payload["graph_sha256"], trace_sha256=payload["trace_sha256"],
            entry_sha256=payload["entry_sha256"], lattice_sha256=lattice.result_sha256,
            signatures_sha256=signatures.result_sha256, signature_identities=signature_ids))
    raw=json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    payload["artifact_sha256"]=hashlib.sha256(raw).hexdigest(); _atomic(output_path,payload)
    _atomic(Path(telemetry_path), {"schema":TELEMETRY_SCHEMA,"acceptance":False,
        "terminal":True,"events":events,"artifact_sha256":payload["artifact_sha256"]})
    return payload


def _load_containment(root):
    import importlib.util
    path=root/".tmp-run-axis0-component-preflight-replacement.py"
    spec=importlib.util.spec_from_file_location("graph_stage_containment",path)
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    return module


def _native_execute_inner(command, *, deadline_seconds, root, stdout_path, stderr_path):
    """Create suspended, contain in the reviewed 8-GiB Job, then resume."""
    import ctypes, msvcrt, subprocess
    from ctypes import wintypes
    c=_load_containment(Path(root)); kernel=c._kernel32(); job=process=thread=None
    stdout=Path(stdout_path).open("xb"); stderr=Path(stderr_path).open("xb"); stdin=open(os.devnull,"rb")
    started=time.monotonic()
    try:
        for stream in (stdin,stdout,stderr): os.set_handle_inheritable(msvcrt.get_osfhandle(stream.fileno()),True)
        job=kernel.CreateJobObjectW(None,None)
        limits=c.JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        limits.BasicLimitInformation.LimitFlags=(c.JOB_OBJECT_LIMIT_PROCESS_MEMORY|c.JOB_OBJECT_LIMIT_JOB_MEMORY|c.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
        limits.ProcessMemoryLimit=8*1024**3; limits.JobMemoryLimit=8*1024**3
        if not kernel.SetInformationJobObject(job,c.JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,ctypes.byref(limits),ctypes.sizeof(limits)): raise ctypes.WinError(ctypes.get_last_error())
        startup=c.STARTUPINFOW(); startup.cb=ctypes.sizeof(startup); startup.dwFlags=c.STARTF_USESTDHANDLES
        startup.hStdInput=wintypes.HANDLE(msvcrt.get_osfhandle(stdin.fileno())); startup.hStdOutput=wintypes.HANDLE(msvcrt.get_osfhandle(stdout.fileno())); startup.hStdError=wintypes.HANDLE(msvcrt.get_osfhandle(stderr.fileno()))
        info=c.PROCESS_INFORMATION(); buffer=ctypes.create_unicode_buffer(subprocess.list2cmdline(command))
        env=dict(os.environ); env["PYTHONPATH"]=str(Path(root)/"src")
        if "--manifest" in command:
            manifest=json.loads(Path(command[command.index("--manifest")+1]).read_text())
            env["MC_PHOTO_GRAPH_STAGE_MANIFEST_SHA256"]=manifest["manifest_sha256"]
        envbuf=ctypes.create_unicode_buffer("\0".join(f"{k}={v}" for k,v in sorted(env.items(),key=lambda x:x[0].upper()))+"\0\0")
        if not kernel.CreateProcessW(command[0],buffer,None,None,True,c.CREATE_SUSPENDED|c.CREATE_UNICODE_ENVIRONMENT,envbuf,str(root),ctypes.byref(startup),ctypes.byref(info)): raise ctypes.WinError(ctypes.get_last_error())
        process,thread=info.hProcess,info.hThread
        if not kernel.AssignProcessToJobObject(job,process): raise ctypes.WinError(ctypes.get_last_error())
        if kernel.ResumeThread(thread)!=1: raise ctypes.WinError(ctypes.get_last_error())
        while True:
            elapsed=time.monotonic()-started
            if elapsed>=deadline_seconds:
                cleanup=c._cleanup_job(kernel,job,process,124); return 124,"deadline_exhausted",cleanup,elapsed
            state=kernel.WaitForSingleObject(process,min(1000,int((deadline_seconds-elapsed)*1000)))
            if state==c.WAIT_OBJECT_0: break
            if state!=c.WAIT_TIMEOUT: raise ctypes.WinError(ctypes.get_last_error())
        exit_code=wintypes.DWORD()
        if not kernel.GetExitCodeProcess(process,ctypes.byref(exit_code)):
            cleanup=c._cleanup_job(kernel,job,process,125)
            return 125,"exit_code_query_failure",cleanup,time.monotonic()-started
        if exit_code.value:
            cleanup=c._cleanup_job(kernel,job,process,127)
            return int(exit_code.value),"child_failure",cleanup,time.monotonic()-started
        if not c._wait_job_empty(kernel,job,5):
            cleanup=c._cleanup_job(kernel,job,process,127)
            return 127,"descendant_survived_root_exit",cleanup,time.monotonic()-started
        return 0,"",{"job_empty":True},time.monotonic()-started
    except Exception as error:
        cleanup=c._cleanup_job(kernel,job,process,125) if 'c' in locals() else {"job_empty":False}
        return 125,"prelaunch_or_containment_failure",{**cleanup,"detail":f"{type(error).__name__}: {error}"},time.monotonic()-started
    finally:
        stdin.close(); stdout.close(); stderr.close()
        for handle in (thread,process,job):
            if handle: kernel.CloseHandle(handle)


def _native_execute(command, *, deadline_seconds, root, stdout_path, stderr_path):
    started=time.monotonic()
    try:
        return _native_execute_inner(command,deadline_seconds=deadline_seconds,root=root,
            stdout_path=stdout_path,stderr_path=stderr_path)
    except Exception as error:
        return (125,"prelaunch_or_containment_failure",
                {"job_empty":False,"detail":f"{type(error).__name__}: {error}"},
                time.monotonic()-started)


def _validate_completed_artifact(value, frozen, project):
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    manifest_pngs=frozen.get("pngs")
    if (not isinstance(manifest_pngs,list)
            or len(manifest_pngs) not in ALLOWED_FRAME_COUNTS
            or not frozen.get("producers")):
        raise ValueError("manifest bindings are empty")
    artifact_inputs=value.get("inputs")
    if (not isinstance(artifact_inputs,list) or
            [(row.get("path"),row.get("sha256")) for row in artifact_inputs] !=
            [(row.get("path"),row.get("sha256")) for row in manifest_pngs]):
        raise ValueError("artifact input order or hashes mismatch manifest")
    loaded=__import__("mc_photogrammetry.project.store",fromlist=["load_project"]).load_project(project)
    if loaded.schema_version!=2: raise ValueError("project schema mismatch")
    if list(map(str,loaded.source_paths)) != [row["path"] for row in manifest_pngs]:
        raise ValueError("project source paths mismatch manifest")
    session=loaded.image_derived_session
    if not isinstance(session,dict): raise ValueError("project session missing")
    session_digest=session.get("session_sha256")
    if not isinstance(session_digest,str) or len(session_digest)!=64:
        raise ValueError("project session digest missing")
    frames=session.get("frames",[])
    if [row.get("source_label") for row in frames] != [row["path"] for row in manifest_pngs]:
        raise ValueError("project session frame binding mismatch")
    # The manifest binds encoded PNG bytes.  Session image_sha256 belongs to a
    # separate domain: shape-framed decoded RGB pixels after the ordinary image
    # loader has applied orientation.  Recompute that digest rather than
    # comparing it with the encoded-file digest.
    from mc_photogrammetry.input.image import load_image
    from mc_photogrammetry.pipeline.image_derived_session import image_sha256
    decoded_digests=[image_sha256(load_image(row["path"])) for row in manifest_pngs]
    if [row.get("image_sha256") for row in frames] != decoded_digests:
        raise ValueError("project session decoded image binding mismatch")
    graph=value.get("graph",{}); payload=graph.get("payload")
    inner=graph.get("result_sha256")
    if (payload is None or graph_mod._sha(payload)!=inner
            or value.get("graph_sha256")!=inner):
        raise ValueError("inner or outer graph digest mismatch")
    if payload.get("session_sha256")!=session_digest:
        raise ValueError("graph and project session digest mismatch")
    if payload.get("terminal")!=graph_mod.GRAPH_TERMINAL:
        raise ValueError("graph terminal mismatch")
    graph_status=payload.get("status")
    assembled=(graph_status=="assembled")
    if (graph_status not in ("assembled","unavailable")
            or value.get("graph_assembled") is not assembled
            or value.get("geometry_available") is not False
            or value.get("solution_available") is not (payload.get("component_solution") is not None)
            or (assembled and (payload.get("reason") or not payload.get("matches")
                               or payload.get("component_solution") is None))
            or (not assembled and (not payload.get("reason") or payload.get("matches")
                                   or payload.get("component_solution") is not None))):
        raise ValueError("graph status or terminal flags are inconsistent")
    identities=value.get("signature_identities")
    if not isinstance(identities,list) or not identities:
        raise ValueError("signature identities missing")
    if [row.get("signature_sha256") for row in identities] != payload.get("signature_digests"):
        raise ValueError("signature identity order mismatch graph")
    trace=value.get("trace")
    if not isinstance(trace,dict): raise ValueError("trace missing")
    trace_digest=trace.get("attribution_sha256"); trace_payload=dict(trace); trace_payload.pop("attribution_sha256",None)
    computed=hashlib.sha256(json.dumps(trace_payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    sources=trace.get("source_digests",{})
    if (trace_digest!=computed or value.get("trace_sha256")!=trace_digest
            or trace.get("graph_result_sha256")!=inner
            or sources.get("session")!=payload.get("session_sha256")
            or sources.get("lattice")!=payload.get("lattice_sha256")
            or sources.get("signatures")!=payload.get("signatures_sha256")):
        raise ValueError("trace digest or provenance join mismatch")
    _validate_solver_failure_observation(value,payload,trace,inner)
    _validate_same_axis_candidates_record(value,trace)


def _validate_solver_failure_observation(value,payload,trace,inner):
    """Validate the optional raw/observer record against its own binding."""
    observation=value.get("solver_failure_observation")
    if observation is None:
        return
    if not isinstance(observation,dict):
        raise ValueError("solver failure observation is malformed")
    from mc_photogrammetry.pipeline import image_derived_graph_attribution as attribution_mod
    body=dict(observation); claimed=body.pop("observation_sha256",None)
    computed=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    raw=observation.get("raw"); binding=observation.get("binding")
    allowed={"schema","frame_count","raw","projected","phase_capture","binding",
             "solver_input_bindings","observation_sha256"}
    if (observation.get("schema")!=OBSERVATION_SCHEMA or claimed!=computed
            or set(observation)-allowed
            or not isinstance(raw,dict)
            or raw.get("schema")!=attribution_mod.SOLVER_FAILURE_OBSERVATION_SCHEMA
            or observation.get("projected")!=trace.get("solver_failure")
            or not isinstance(binding,dict)):
        raise ValueError("solver failure observation replay mismatch")
    # DS-IMG-SOLVER-CONTEXT-CAPTURE-DIAGNOSTIC-20261004: replay the optional
    # exact solve-call context fail closed and bind it to the same failed
    # snapshot's own frame universe. The key is absent for historical artifacts
    # and for any failure snapshot that never built the capture, so old records
    # still validate without it and R10 stays an incomplete-context artifact.
    solve_context=raw.get("solve_context")
    if solve_context is not None:
        from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
        graph_mod.validate_solver_solve_context(solve_context, raw.get("frames"))
    phase_capture=observation.get("phase_capture")
    if phase_capture is not None:
        # A present phase capture must replay through its own fail-closed
        # validator and stay bound to the same failed assignment as the
        # active-edge alias record it accompanies. The alias record is carried
        # by the projected failure trace (the raw snapshot never held it), so a
        # raw lookup would silently skip this binding.
        from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
        graph_mod.validate_solver_phase_capture(phase_capture)
        projected_failure=trace.get("solver_failure")
        raw_aliases=(projected_failure.get("active_edge_aliases")
                     if isinstance(projected_failure,dict) else None)
        if (phase_capture.get("status")=="complete"
                and isinstance(raw_aliases,dict)
                and raw_aliases.get("status")=="complete"
                and phase_capture.get("source_sha256")
                != raw_aliases.get("failed_assignment_sha256")):
            raise ValueError("phase capture source mismatch")
    solver_input_bindings=observation.get("solver_input_bindings")
    if solver_input_bindings is not None:
        # DS-R7: a present solver-input binding record must replay through its
        # own fail-closed validator and stay bound to the same failed assignment
        # as the active-edge alias record.
        from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
        graph_mod.validate_solver_input_bindings(solver_input_bindings)
        projected_failure=trace.get("solver_failure")
        raw_aliases=(projected_failure.get("active_edge_aliases")
                     if isinstance(projected_failure,dict) else None)
        if (solver_input_bindings.get("status")=="complete"
                and isinstance(raw_aliases,dict)
                and raw_aliases.get("status")=="complete"
                and solver_input_bindings.get("failed_assignment_sha256")
                != raw_aliases.get("failed_assignment_sha256")):
            raise ValueError("solver input bindings assignment mismatch")
        if (solver_input_bindings.get("status")=="complete"
                and isinstance(raw_aliases,dict)
                and raw_aliases.get("status")=="complete"
                and solver_input_bindings.get("input_match_identity_sha256")
                != raw_aliases.get("input_match_identity_sha256")):
            raise ValueError("solver input bindings input identity mismatch")
    if (binding.get("entry_sha256")!=value.get("entry_sha256")
            or binding.get("graph_sha256")!=inner
            or binding.get("lattice_sha256")!=payload.get("lattice_sha256")
            or binding.get("signatures_sha256")!=payload.get("signatures_sha256")
            or binding.get("input_sha256")!=[row.get("sha256") for row in value.get("inputs",[])]):
        raise ValueError("solver failure observation binding mismatch")


def _validate_same_axis_candidates_record(value,trace):
    """Replay the optional pre-filter candidate record and its shared bindings.

    DS-IMG-SAMEAXIS-PREFILTER-RUNNER-WIRING-R10 /
    DS-IMG-SAMEAXIS-R10-BINDING-REPAIR. A present record must always replay
    through its own fail-closed validator. A ``status=complete`` record must
    additionally name the same failed best assignment and the same ordered
    post-filter solver-input identity as the projected active-edge alias record,
    which is the retained location of both shared digests. That binding is
    required, not optional: a complete record whose projected solver failure is
    absent or not a mapping, or whose alias binding is absent, not a mapping, or
    not ``status=complete``, is rejected instead of silently skipping both digest
    checks. Only then are both shared digests compared. A typed refusal record is
    retained verbatim and carries no digests, so only its own replay applies.

    DS-IMG-SAMEAXIS-ALLAXIS-CAPTURE-V2: the replay accepts the v1 single-axis
    record and the v2 all-axis aggregate, which share the same two digest
    domains, so this fail-closed cross-binding is retained unchanged for both.
    """
    record=value.get("same_axis_candidates")
    if record is None:
        return
    if not isinstance(record,dict):
        raise ValueError("same axis candidates record is malformed")
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    graph_mod.validate_same_axis_candidates(record)
    if record.get("status")!="complete":
        return
    projected_failure=trace.get("solver_failure")
    if not isinstance(projected_failure,dict):
        raise ValueError("same axis candidates require a projected solver failure")
    raw_aliases=projected_failure.get("active_edge_aliases")
    if not isinstance(raw_aliases,dict) or raw_aliases.get("status")!="complete":
        raise ValueError("same axis candidates require a complete alias binding")
    if record.get("failed_assignment_sha256") != raw_aliases.get(
            "failed_assignment_sha256"):
        raise ValueError("same axis candidates assignment mismatch")
    if record.get("input_match_identity_sha256") != raw_aliases.get(
            "input_match_identity_sha256"):
        raise ValueError("same axis candidates input identity mismatch")


def run_child(command, *, deadline_seconds, terminal_path, heartbeat_path,
              manifest_path, stdout_path, stderr_path, native_execute=_native_execute):
    manifest=Path(manifest_path)
    if not manifest.is_file(): raise ValueError("frozen manifest is required")
    frozen=json.loads(manifest.read_text()); claimed=frozen.pop("manifest_sha256",None)
    computed=hashlib.sha256(json.dumps(frozen,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if claimed!=computed: raise ValueError("manifest digest mismatch")
    if frozen.get("schema")!="mc-photogrammetry-evaluation-graph-stage-manifest-v1":
        raise ValueError("manifest schema mismatch")
    if len(frozen.get("pngs",[])) not in ALLOWED_FRAME_COUNTS or not frozen.get("producers"):
        raise ValueError("manifest bindings are empty")
    for row in tuple(frozen.get("pngs",()))+tuple(frozen.get("producers",())):
        path=Path(row["path"])
        if not path.is_file() or _sha(path)!=row["sha256"]:
            raise ValueError("manifest input drift")
    if (frozen.get("limit_bytes")!=8*1024**3
            or frozen.get("deadline_seconds")!=deadline_seconds):
        raise ValueError("manifest resource contract mismatch")
    if frozen.get("worker_command") and frozen["worker_command"]!=list(command):
        raise ValueError("worker command mismatch manifest")
    code,reason,cleanup,elapsed=native_execute(command,deadline_seconds=deadline_seconds,
        root=Path(__file__).resolve().parents[1],stdout_path=stdout_path,stderr_path=stderr_path)
    artifact=Path(frozen["output_path"])
    geometry_available=False
    artifact_execution_complete=False; artifact_reason=""; artifact_refusal=None
    if code==0:
        try:
            value=json.loads(artifact.read_text()); claimed_artifact=value.pop("artifact_sha256")
            computed_artifact=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
            project=Path(frozen["project_path"])
            if (value.get("schema")!=SCHEMA or claimed_artifact!=computed_artifact
                    or (value.get("project_sha256") and
                        (not project.is_file() or value.get("project_sha256")!=_sha(project)))):
                raise ValueError("artifact or project digest mismatch")
            if value.get("execution_complete"):
                if not project.is_file(): raise ValueError("completed stage lacks project")
                source_root=str(Path(__file__).resolve().parents[1]/"src")
                if source_root not in sys.path: sys.path.insert(0,source_root)
                _validate_completed_artifact(value,frozen,project)
            geometry_available=bool(value.get("geometry_available"))
            artifact_execution_complete=bool(value.get("execution_complete"))
            artifact_reason=str(value.get("reason", ""))
            artifact_refusal=value.get("refusal")
            if artifact_refusal is not None:
                from mc_photogrammetry.reconstruction import plane_placement as placement
                work_record=artifact_refusal.get("work")
                work=None if work_record is None else placement.PlacementWork(**work_record)
                replay=placement.PlacementRefusal(artifact_refusal["reason"],artifact_refusal["detail"],
                    artifact_refusal["detail_sha256"],work,artifact_refusal["refusal_sha256"])
                if replay.to_record()!=artifact_refusal or artifact_reason!=replay.reason:
                    raise ValueError("typed refusal artifact mismatch")
        except Exception:
            code=126; reason="missing_or_invalid_artifact"
    heartbeat={}
    if Path(heartbeat_path).is_file():
        try: heartbeat=json.loads(Path(heartbeat_path).read_text())
        except Exception: heartbeat={}
    events=heartbeat.get("events",[]) if isinstance(heartbeat,dict) else []
    started_stages=[row.get("stage") for row in events if row.get("state")=="started"]
    completed_stages=[row.get("stage") for row in events if row.get("state") in ("complete","estimated")]
    terminal={"schema":TELEMETRY_SCHEMA,"acceptance":False,"terminal":True,
        "status":"complete" if code==0 and artifact_execution_complete else "refused",
        "reason":reason or artifact_reason,"exit_code":code,
        "refusal":artifact_refusal,
        "execution_complete":code==0 and artifact_execution_complete,
        "geometry_available":geometry_available,"elapsed_seconds":elapsed,
        "manifest_sha256":claimed,"cleanup":cleanup,
        "last_started_stage":started_stages[-1] if started_stages else "",
        "last_completed_stage":completed_stages[-1] if completed_stages else "",
        "rss_bytes":events[-1].get("rss_bytes") if events else None,
        "heartbeat_sha256":_sha(Path(heartbeat_path)) if Path(heartbeat_path).is_file() else ""}
    _atomic(Path(terminal_path),terminal); return code


def create_manifest(*, png_paths, expected_sha256, project_path, output_path,
                    telemetry_path, terminal_path, manifest_path, stdout_path, stderr_path,
                    deadline_seconds=600, worker_command=()):
    root=Path(__file__).resolve().parents[1]
    inputs=tuple(Path(p).resolve() for p in png_paths)
    outputs=tuple(Path(p).resolve() for p in (project_path,output_path,telemetry_path,
                                               terminal_path,manifest_path,stdout_path,stderr_path))
    if len(set(outputs))!=len(outputs) or set(inputs)&set(outputs):
        raise ValueError("output paths collide or alias inputs")
    if any(path.exists() for path in outputs): raise ValueError("fresh output paths are required")
    if len(inputs) not in ALLOWED_FRAME_COUNTS or len(expected_sha256)!=len(inputs):
        raise ValueError("three or four PNG bindings required")
    for path,digest in zip(inputs,expected_sha256):
        if not path.is_file() or _sha(path)!=digest: raise ValueError("PNG manifest binding mismatch")
    producers=[Path(__file__),root/"src/mc_photogrammetry/pipeline/image_derived_entry.py",
        root/"src/mc_photogrammetry/pipeline/image_derived_lattice_measure.py",
        root/"src/mc_photogrammetry/pipeline/image_derived_signatures.py",
        root/"src/mc_photogrammetry/pipeline/image_derived_graph.py",
        root/"src/mc_photogrammetry/pipeline/image_derived_graph_attribution.py",
        root/".tmp-run-axis0-component-preflight-replacement.py",
        root/"src/mc_photogrammetry/reconstruction/plane_graph.py",
        root/"src/mc_photogrammetry/reconstruction/lattice_planes.py"]
    value={"schema":"mc-photogrammetry-evaluation-graph-stage-manifest-v1",
        "pngs":[{"path":str(p),"sha256":d} for p,d in zip(inputs,expected_sha256)],
        "producers":[{"path":str(p.resolve()),"sha256":_sha(p)} for p in producers],
        "project_path":str(outputs[0]),"output_path":str(outputs[1]),
        "telemetry_path":str(outputs[2]),"terminal_path":str(outputs[3]),
        "stdout_path":str(outputs[5]),"stderr_path":str(outputs[6]),
        "worker_command":list(worker_command),
        "limit_bytes":8*1024**3,"deadline_seconds":deadline_seconds}
    raw=json.dumps(value,sort_keys=True,separators=(",",":")).encode(); value["manifest_sha256"]=hashlib.sha256(raw).hexdigest()
    _atomic(Path(manifest_path),value); return value


def _verify_worker_context(args):
    import ctypes
    from ctypes import wintypes
    if not args.manifest: raise ValueError("worker requires frozen manifest")
    manifest=json.loads(Path(args.manifest).read_text()); claimed=manifest.pop("manifest_sha256",None)
    computed=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if claimed!=computed or os.environ.get("MC_PHOTO_GRAPH_STAGE_MANIFEST_SHA256")!=claimed:
        raise ValueError("worker manifest or containment token mismatch")
    bindings=[(str(Path(p).resolve()),d) for p,d in zip(args.png,args.sha256)]
    if bindings != [(row["path"],row["sha256"]) for row in manifest.get("pngs",[])]:
        raise ValueError("worker PNG command mismatch manifest")
    expected=(str(Path(args.project).resolve()),str(Path(args.output).resolve()),str(Path(args.telemetry).resolve()))
    if expected!=(manifest["project_path"],manifest["output_path"],manifest["telemetry_path"]):
        raise ValueError("worker output command mismatch manifest")
    actual_command=[sys.executable,str(Path(__file__).resolve())]+sys.argv[1:]
    if actual_command!=manifest.get("worker_command"):
        raise ValueError("worker argv mismatch manifest")
    kernel=ctypes.WinDLL("kernel32",use_last_error=True); kernel.GetCurrentProcess.restype=wintypes.HANDLE
    kernel.IsProcessInJob.argtypes=[wintypes.HANDLE,wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL)]; kernel.IsProcessInJob.restype=wintypes.BOOL
    contained=wintypes.BOOL()
    if not kernel.IsProcessInJob(kernel.GetCurrentProcess(),None,ctypes.byref(contained)) or not contained.value:
        raise ValueError("worker is not contained in a Job")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--png",action="append",required=True)
    parser.add_argument("--sha256",action="append",required=True); parser.add_argument("--project",required=True)
    parser.add_argument("--output",required=True); parser.add_argument("--telemetry",required=True)
    parser.add_argument("--contained",action="store_true"); parser.add_argument("--terminal")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--manifest"); parser.add_argument("--stdout"); parser.add_argument("--stderr")
    args=parser.parse_args()
    if args.contained:
        if not all((args.terminal,args.manifest,args.stdout,args.stderr)): parser.error("contained paths required")
        child=[sys.executable,str(Path(__file__).resolve()),"--worker","--manifest",args.manifest]
        for p in args.png: child.extend(("--png",p))
        for d in args.sha256: child.extend(("--sha256",d))
        child.extend(("--project",args.project,"--output",args.output,"--telemetry",args.telemetry))
        create_manifest(png_paths=args.png,expected_sha256=args.sha256,project_path=args.project,
            output_path=args.output,telemetry_path=args.telemetry,terminal_path=args.terminal,
            manifest_path=args.manifest,stdout_path=args.stdout,stderr_path=args.stderr,
            worker_command=child)
        raise SystemExit(run_child(child,deadline_seconds=600,terminal_path=args.terminal,
            heartbeat_path=args.telemetry,manifest_path=args.manifest,
            stdout_path=args.stdout,stderr_path=args.stderr))
    if not args.worker:
        parser.error("CLI execution requires --contained")
    _verify_worker_context(args)
    execute_graph_stage(png_paths=args.png,expected_sha256=args.sha256,project_path=args.project,
        output_path=args.output,telemetry_path=args.telemetry)


if __name__=="__main__": main()
