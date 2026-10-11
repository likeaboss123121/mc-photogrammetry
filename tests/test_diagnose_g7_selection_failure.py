"""Synthetic-only G7 observation and no-clobber diagnostic publication."""
import dataclasses
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
from mc_photogrammetry.pipeline import image_derived_joint_graph as graph_mod
from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp, plane_graph
from test_endpoint_hypothesis import connected_fixture
from test_image_derived_endpoint_selection import _evidence, _forge
from test_image_derived_graph_v3 import _source_records

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluation/diagnose_g7_selection_failure.py"
spec = importlib.util.spec_from_file_location("g7_diagnostic_test_subject", SCRIPT)
subject = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = subject
spec.loader.exec_module(subject)


def controlled_graph(kind="no_active_factors"):
    readings, rows = connected_fixture("a", "b")
    budget = csp.EndpointBudget()
    if kind == "no_active_factors":
        rows = []
    elif kind == "unsupported_reading_endpoint":
        rows = rows[:2]
    elif kind == "axis_2_disconnected":
        readings = [r for r in readings if r.axis != 2]
        rows = rows[:2]
    elif kind == "budget_refused":
        budget = dataclasses.replace(budget, branch_nodes=0)
    else:
        raise AssertionError(kind)
    readings, rows = tuple(readings), tuple(rows)
    signatures = _evidence(readings)
    sources = dict(session=SimpleNamespace(session_sha256="a"*64),
                   lattice_result=SimpleNamespace(result_sha256="b"*64, readings=readings),
                   signatures_result=SimpleNamespace(result_sha256="c"*64, signatures=signatures),
                   connection_graph=None)
    cert = selection.select_endpoint_hypotheses(readings=readings, signatures=signatures, candidate_matches=rows,
        frame_ids=("a", "b"), quadrant_constraints=(("a", "b", 0),), budget=budget,
        session_sha256="a"*64,lattice_sha256="b"*64,signatures_sha256="c"*64)
    graph = graph_mod.ImageDerivedGraphResult(graph_mod.IMAGE_DERIVED_GRAPH_SCHEMA,graph_mod.IMAGE_DERIVED_GRAPH_PRODUCER,
        graph_mod.GRAPH_TERMINAL,"unavailable",cert.reason,cert.status,"a"*64,"b"*64,"c"*64,"",("a","b"),
        readings,signatures,rows,cert)
    return graph,sources


def capture_for(graph,sources):
    capture=subject.GraphCapture()
    capture.calls=1;capture.graph=graph;capture.arguments=((),sources)
    return capture


def _run(tmp_path,monkeypatch,graph=None,sources=None,*,delivery=None):
    if delivery is None:
        if graph is None: graph,sources=controlled_graph()
        monkeypatch.setattr(subject.orchestration.g_mod,"derive_image_derived_graph",lambda **kwargs:graph)
        def delivery(images,**kwargs):
            observed=subject.orchestration.g_mod.derive_image_derived_graph(**sources)
            Path(kwargs["project_path"]).write_bytes(b"synthetic production output")
            return observed
    image=tmp_path/"ordinary.png";image.write_bytes(b"synthetic input bytes; never decoded")
    return subject.run_once([image],project=tmp_path/"out.mcscan",destination=tmp_path/"out.litematic",
        diagnostic=tmp_path/"diagnostic.json",name="synthetic",description="synthetic",delivery=delivery)


@pytest.mark.parametrize("kind",["no_active_factors","unsupported_reading_endpoint","axis_2_disconnected","budget_refused"])
def test_real_certificate_failure_reasons_accounting_and_roundtrip(kind,tmp_path,monkeypatch):
    graph,sources=controlled_graph(kind)
    monkeypatch.setattr(plane_graph,"match_plane_graph",lambda *a,**k:pytest.fail("second matcher"))
    monkeypatch.setattr(selection,"select_endpoint_hypotheses",lambda *a,**k:pytest.fail("second selector"))
    monkeypatch.setattr(plane_graph,"_evaluate_assignment",lambda *a,**k:pytest.fail("second evaluator"))
    result=_run(tmp_path,monkeypatch,graph,sources)
    assert result.production_result is graph and result.production_error is None and result.diagnostic_error is None
    record=json.loads((tmp_path/"diagnostic.json").read_bytes())
    subject.validate_report(record)
    cert=subject._certificate_from_record(record["graph"]["selection"])
    assert cert.result_sha256==graph.selection.result_sha256
    counts=cert.proof_record()["counters"]
    assert counts["eligible_assignments"]+counts["excluded_assignments"]+counts["eligibility_unclassified_assignments"]==counts["visited_assignments"]
    assert counts["visited_assignments"]+counts["unvisited_assignments"]==counts["planned_assignments"]
    if kind=="budget_refused":
        assert record["status"]=="refused" and record["graph"]["proof_scope"]=="partial"
        assert not cert.proof_record()["exhaustive"] and cert.proof_record()["denial"]
    else:
        assert record["status"]=="complete" and cert.proof_record()["eligibility_reasons"]=={kind:1}
        assert record["graph"]["geometry_evaluations"]==0
    assert (tmp_path/"out.mcscan").read_bytes()==b"synthetic production output"
    assert not (tmp_path/"out.litematic").exists()
    assert len(record["graph"]["endpoint_incidence"])==len(graph.readings)


def test_source_backed_success_is_compact_no_failure_and_one_pass(monkeypatch,tmp_path):
    sources=_source_records(monkeypatch)
    original=subject.orchestration.g_mod.derive_image_derived_graph
    counts=dict(matcher=0,selector=0,evaluator=0)
    for key,module,name in [("matcher",plane_graph,"match_plane_graph"),("selector",selection,"select_endpoint_hypotheses"),("evaluator",plane_graph,"_evaluate_assignment")]:
        saved=getattr(module,name)
        def counted(*args,_key=key,_saved=saved,**kwargs):
            counts[_key]+=1
            return _saved(*args,**kwargs)
        monkeypatch.setattr(module,name,counted)
    def delivery(images,**kwargs):
        graph=subject.orchestration.g_mod.derive_image_derived_graph(**sources)
        Path(kwargs["project_path"]).write_bytes(graph.result_sha256.encode())
        return graph
    off=delivery((),project_path=tmp_path/"off.mcscan")
    assert counts==dict(matcher=1,selector=1,evaluator=1)
    on=_run(tmp_path,monkeypatch,delivery=delivery)
    assert counts==dict(matcher=2,selector=2,evaluator=2)
    assert off.status=="assembled" and on.production_result.status=="assembled"
    assert off.result_sha256==on.production_result.result_sha256
    assert (tmp_path/"off.mcscan").read_bytes()==(tmp_path/"out.mcscan").read_bytes()
    assert on.report["status"]=="no_failure" and on.diagnostic_error is None
    assert set(on.report["graph"]["selection"])=={"schema","producer","status","result_sha256"}
    assert "proof" not in json.dumps(on.report["graph"])
    assert subject.orchestration.g_mod.derive_image_derived_graph is original


def test_geometry_refusal_preserves_reconstructable_representatives(monkeypatch,tmp_path):
    sources=_source_records(monkeypatch)
    saved=graph_mod._evaluate_selected;calls=[]
    def refused(*args):
        calls.append(1)
        return dataclasses.replace(saved(*args),reason="plane_integer_residual")
    monkeypatch.setattr(graph_mod,"_evaluate_selected",refused)
    graph=graph_mod.derive_image_derived_graph(**sources)
    assert graph.status=="unavailable" and graph.selection.status=="unique" and calls==[1]
    result=_run(tmp_path,monkeypatch,graph,sources)
    assert result.report["status"]=="complete" and result.diagnostic_error is None and calls==[1]
    stored=json.loads((tmp_path/"diagnostic.json").read_bytes())
    rebuilt=subject._certificate_from_record(stored["graph"]["selection"])
    assert selection._payload(rebuilt)==selection._payload(graph.selection)
    assert rebuilt.result_sha256==graph.selection.result_sha256
    stored["graph"]["selection"]["representatives"][0][2][8]+=1.
    stored.pop("result_sha256");subject._seal(stored)
    with pytest.raises(subject.DiagnosticError):subject.validate_report(stored)


def test_actual_default_delivery_driver_produces_failure_record(monkeypatch,tmp_path):
    from mc_photogrammetry.pipeline import image_derived_production as production
    from mc_photogrammetry.pipeline import image_derived_entry as entry_mod
    from mc_photogrammetry.pipeline import image_derived_pose_graph as pose_mod
    import test_image_derived_graph_v3 as fixture_mod
    original_session=fixture_mod._simple_session
    def labelled_session(patch,frame_ids):
        session=original_session(patch,frame_ids)
        frames=tuple(dataclasses.replace(f,source_label=str(tmp_path/(f.frame_id+".png"))) for f in session.frames)
        return dataclasses.replace(session,frames=frames,input_frames=frames,session_sha256=None)
    monkeypatch.setattr(fixture_mod,"_simple_session",labelled_session)
    sources=_source_records(monkeypatch)
    connection=sources["connection_graph"];session=sources["session"]
    pose=pose_mod.compose_image_derived_pose_graph(connection,budgets=pose_mod.PoseGraphBudgets(128,1000,1.,.1))
    assert pose.status=="ok"
    reason,detail=entry_mod._terminal_reason(connection,pose)
    entry=entry_mod.ImageDerivedEntryResult(entry_mod.IMAGE_DERIVED_ENTRY_SCHEMA,entry_mod.IMAGE_DERIVED_ENTRY_PRODUCER,
        entry_mod.GEOMETRY_UNAVAILABLE,reason,detail,str(tmp_path/"out.mcscan"),tuple(f.source_label for f in session.input_frames),
        connection.budgets,pose.budgets,session,connection,pose)
    monkeypatch.setattr(production.entry_mod,"build_image_derived_entry",lambda *args,**kwargs:entry)
    monkeypatch.setattr(production.lattice_mod,"measure_image_derived_lattice",lambda **kwargs:sources["lattice_result"])
    monkeypatch.setattr(subject.orchestration.f1_mod,"derive_image_derived_signatures",lambda **kwargs:sources["signatures_result"])
    calls=[]
    def matcher(*args, **kwargs):
        calls.append(1)
        kwargs["admitted_candidate_sink"].append(plane_graph.AdmittedCandidateUniverse((), (), ()))
        return ()
    monkeypatch.setattr(plane_graph,"match_plane_graph",matcher)
    monkeypatch.setattr(plane_graph,"_evaluate_assignment",lambda *a,**k:pytest.fail("no-factor evaluation"))
    images=tuple(tmp_path/(f.frame_id+".png") for f in session.frames)
    for image in images:image.write_bytes(b"synthetic bytes; entry/estimators replaced with accepted synthetic records")
    result=subject.run_once(images,project=tmp_path/"out.mcscan",destination=tmp_path/"out.litematic",
        diagnostic=tmp_path/"diagnostic.json",name="synthetic",description="synthetic")
    assert calls==[1] and result.production_error is None and result.diagnostic_error is None
    assert result.production_result.status=="geometry_unavailable" and result.production_result.reason=="g_unavailable"
    assert result.report["status"]=="complete" and result.report["g_calls"]==1
    proof=result.report["graph"]["selection"]["proof"]
    assert proof["eligibility_reasons"]=={"no_active_factors":1}
    assert (tmp_path/"out.mcscan").exists() and not (tmp_path/"out.litematic").exists()


def test_resealed_complete_receipt_cannot_omit_ordered_inputs(tmp_path,monkeypatch):
    result=_run(tmp_path,monkeypatch)
    result.report["run"]["inputs"]=[];result.report.pop("result_sha256");subject._seal(result.report)
    with pytest.raises(subject.DiagnosticError):subject.validate_report(result.report)


def test_wrapper_forwards_objects_and_returns_identical_result_and_restores(monkeypatch):
    arguments=(object(),object());token=object();returned=object();seen=[]
    def original(*args,**kwargs):
        seen.append((args,kwargs));return returned
    monkeypatch.setattr(subject.orchestration.g_mod,"derive_image_derived_graph",original)
    capture=subject.GraphCapture()
    def delivery():return subject.orchestration.g_mod.derive_image_derived_graph(*arguments,token=token)
    assert capture.invoke(delivery) is returned and capture.graph is returned and capture.calls==1
    assert all(a is b for a,b in zip(seen[0][0],arguments)) and seen[0][1]["token"] is token
    assert subject.orchestration.g_mod.derive_image_derived_graph is original


def test_wrapper_restoration_and_exact_exception(monkeypatch,tmp_path):
    error=RuntimeError("synthetic");original=lambda **kwargs:(_ for _ in ()).throw(error)
    monkeypatch.setattr(subject.orchestration.g_mod,"derive_image_derived_graph",original)
    def delivery(*args,**kwargs):return subject.orchestration.g_mod.derive_image_derived_graph()
    result=_run(tmp_path,monkeypatch,delivery=delivery)
    assert result.production_error is error and result.report["reason"]=="production_exception"
    assert subject.orchestration.g_mod.derive_image_derived_graph is original


@pytest.mark.parametrize("calls",[0,2])
def test_zero_multiple_calls_are_incomplete_without_rerun(calls,monkeypatch,tmp_path):
    graph,_=controlled_graph();count=[]
    def original():count.append(1);return graph
    monkeypatch.setattr(subject.orchestration.g_mod,"derive_image_derived_graph",original)
    terminal=object()
    def delivery(*args,**kwargs):
        for _ in range(calls):subject.orchestration.g_mod.derive_image_derived_graph()
        return terminal
    result=_run(tmp_path,monkeypatch,delivery=delivery)
    assert result.production_result is terminal and len(count)==calls
    assert result.report["status"]=="incomplete" and result.report["reason"]=="g_call_count"
    assert result.report["g_calls"]==calls


@pytest.mark.parametrize("change",["foreign_graph","foreign_cert","graph_producer","cert_producer","stale_binding","proof_size","accounting","proof_digest"])
def test_foreign_malformed_stale_oversized_proofs_never_complete(change,tmp_path,monkeypatch):
    graph,sources=controlled_graph()
    if change=="foreign_graph":graph=object()
    elif change=="foreign_cert":graph=_forge(graph,selection=object())
    elif change=="graph_producer":graph=_forge(graph,producer="m9-g-image-derived-graph-6")
    elif change=="cert_producer":graph=_forge(graph,selection=_forge(graph.selection,producer="old"))
    elif change=="stale_binding":sources=dict(sources,session=SimpleNamespace(session_sha256="f"*64))
    elif change=="proof_size":graph=_forge(graph,selection=_forge(graph.selection,proof_json="x"*(subject.MAX_PROOF_BYTES+1)))
    else:
        proof=graph.selection.proof_record()
        if change=="accounting":proof["counters"]["excluded_assignments"]+=1
        else:proof["eligibility_digest"]="nothex"
        graph=_forge(graph,selection=_forge(graph.selection,proof_json=selection._encode(proof).decode()))
    result=_run(tmp_path,monkeypatch,graph,sources)
    assert result.report["status"]=="incomplete" and result.production_result is graph
    assert (tmp_path/"out.mcscan").exists()


@pytest.mark.parametrize("value",[{}, {"kind":"failure"}])
def test_resealed_complete_empty_graph_cannot_pass(value):
    report=subject._seal(dict(schema=subject.SCHEMA,producer=subject.PRODUCER,status="complete",reason="",field="",g_calls=1,graph=value,run={},notes=[]))
    with pytest.raises(subject.DiagnosticError):subject.validate_report(report)


@pytest.mark.parametrize("value",["x"*(subject.MAX_REPORT_BYTES+1),2**10000,float("inf"),object()],ids=["oversized_string","oversized_integer","nonfinite","object"])
def test_scalar_preflight_rejects_before_json_encoder(value,monkeypatch):
    monkeypatch.setattr(subject.json,"JSONEncoder",lambda *a,**k:pytest.fail("encoder must not allocate"))
    with pytest.raises(subject.DiagnosticError):subject.canonical_bytes(value)


def test_image_iterable_stops_at_33_before_any_hash_or_delivery(tmp_path,monkeypatch):
    def images():
        for _ in range(33):yield tmp_path/"fake.png"
        pytest.fail("consumed 34th input")
    monkeypatch.setattr(subject,"_file_hash",lambda *a:pytest.fail("predecode hash"))
    with pytest.raises(subject.DiagnosticError,match="image_count"):
        subject.run_once(images(),project=tmp_path/"p",destination=tmp_path/"l",diagnostic=tmp_path/"d",name="n",description="d")


@pytest.mark.parametrize("mode",["mutated","missing"])
def test_postrun_input_change_preserves_outputs_and_prevents_complete(mode,tmp_path,monkeypatch):
    terminal=object()
    def delivery(images,**kwargs):
        Path(kwargs["project_path"]).write_bytes(b"production must survive")
        if mode=="mutated":images[0].write_bytes(b"changed")
        else:images[0].unlink()
        return terminal
    result=_run(tmp_path,monkeypatch,delivery=delivery)
    assert result.production_result is terminal and result.report["reason"]=="input_changed"
    assert result.report["status"]=="refused" and not (tmp_path/"diagnostic.json").exists()
    assert (tmp_path/"out.mcscan").read_bytes()==b"production must survive"


def bare_report():
    return subject.build_report(subject.GraphCapture(),{})


def test_atomic_publish_visibility_and_readback(tmp_path,monkeypatch):
    target=tmp_path/"diagnostic.json";report=bare_report();real_fsync=subject.os.fsync;real_link=subject.os.link;seen=[]
    def fsync(fd):
        assert not target.exists();real_fsync(fd);seen.append("fsync")
    def link(source,destination):
        assert seen==["fsync"] and not target.exists()
        assert subject._readback(source)==subject.canonical_bytes(report)
        real_link(source,destination);seen.append("link")
    monkeypatch.setattr(subject.os,"fsync",fsync);monkeypatch.setattr(subject.os,"link",link)
    subject.publish_report(target,report)
    assert seen==["fsync","link"] and subject._readback(target)==subject.canonical_bytes(report)
    assert list(tmp_path.iterdir())==[target]


@pytest.mark.parametrize("failure",["collision","race","write","fsync","unsupported_link","temporary_readback","final_readback"])
def test_publication_failures_are_atomic_and_preserve_foreign_target(failure,tmp_path,monkeypatch):
    target=tmp_path/"diagnostic.json"
    if failure=="collision":target.write_bytes(b"foreign")
    elif failure=="race":
        def link(source,destination):
            target.write_bytes(b"foreign");raise FileExistsError()
        monkeypatch.setattr(subject.os,"link",link)
    elif failure in ("fsync","unsupported_link"):
        def fail(*args):raise OSError("synthetic")
        monkeypatch.setattr(subject.os,"fsync" if failure=="fsync" else "link",fail)
    elif failure=="write":
        real=subject.os.fdopen
        class FailedWrite:
            def __init__(self,fd):self.stream=real(fd,"wb")
            def __enter__(self):return self
            def __exit__(self,*args):self.stream.close()
            def write(self,data):self.stream.write(data[:5]);raise OSError("partial write")
        monkeypatch.setattr(subject.os,"fdopen",lambda fd,*a:FailedWrite(fd))
    else:
        real=subject._readback
        def readback(path):
            if (Path(path)==target)==(failure=="final_readback"):raise subject.DiagnosticError("readback_failed","synthetic")
            return real(path)
        monkeypatch.setattr(subject,"_readback",readback)
    with pytest.raises(subject.DiagnosticError):subject.publish_report(target,bare_report())
    assert not list(tmp_path.glob(".g7-diagnostic-*.tmp"))
    if failure in ("collision","race"):assert target.read_bytes()==b"foreign"
    else:assert not target.exists()


def test_diagnostic_write_failure_cannot_change_production_output_or_result(tmp_path,monkeypatch):
    def failed(*args):raise subject.DiagnosticError("publication_failed","synthetic")
    monkeypatch.setattr(subject,"publish_report",failed)
    graph,sources=controlled_graph()
    result=_run(tmp_path,monkeypatch,graph,sources)
    assert result.production_result is graph and result.production_error is None
    assert result.diagnostic_error.reason=="publication_failed"
    assert (tmp_path/"out.mcscan").read_bytes()==b"synthetic production output"
    assert not (tmp_path/"diagnostic.json").exists()


@pytest.mark.parametrize("flag",["--max-frames","--camera-manifest","--bounds","--palette","--budget","--overwrite"])
def test_cli_has_no_metadata_or_resource_passthrough(flag):
    with pytest.raises(SystemExit):subject.main(["--image","a.png","--project","p","--destination","l","--diagnostic","d","--name","n","--description","d",flag,"x"])


@pytest.mark.parametrize("limit",["depth","items"])
def test_proof_predecode_limit_never_enters_decoder_or_graph_constructor(limit,monkeypatch):
    graph,sources=controlled_graph()
    if limit=="depth":
        raw="["*(subject.MAX_DEPTH+2)+"0"+"]"*(subject.MAX_DEPTH+2)
    else:
        monkeypatch.setattr(subject,"MAX_ITEMS",64)
        raw="["+",".join("0" for _ in range(65))+"]"
    forged=_forge(graph,selection=_forge(graph.selection,proof_json=raw))
    monkeypatch.setattr(subject.json,"loads",lambda *a,**k:pytest.fail("decoder must not enter"))
    # Keep the exact isinstance/type guard class; trap constructor separately.
    monkeypatch.setattr(graph_mod.ImageDerivedGraphResult,"__post_init__",lambda *a:pytest.fail("constructor must not enter"))
    with pytest.raises(subject.DiagnosticError,match="json_preflight: "+limit):
        subject._graph_record(capture_for(forged,sources))


@pytest.mark.parametrize("limit",["depth","items"])
def test_readback_predecode_limit_never_enters_decoder(limit,tmp_path,monkeypatch):
    if limit=="depth":raw="["*(subject.MAX_DEPTH+2)+"0"+"]"*(subject.MAX_DEPTH+2)
    else:
        monkeypatch.setattr(subject,"MAX_ITEMS",64)
        raw="["+",".join("0" for _ in range(65))+"]"
    target=tmp_path/"hostile.json";target.write_text(raw)
    monkeypatch.setattr(subject.json,"loads",lambda *a,**k:pytest.fail("decoder must not enter"))
    with pytest.raises(subject.DiagnosticError,match="json_preflight: "+limit):subject._readback(target)


@pytest.mark.parametrize("char,count",[("\x00",12),("\n",32),("\x7f",12),("\u00e9",12),("\U0001f600",6)])
@pytest.mark.parametrize("as_key",[False,True])
def test_escaped_utf8_precharge_refuses_before_encoder(char,count,as_key,monkeypatch):
    value=char*count
    assert len(value.encode("utf-8"))<64
    monkeypatch.setattr(subject,"MAX_REPORT_BYTES",64)
    monkeypatch.setattr(subject.json,"JSONEncoder",lambda *a,**k:pytest.fail("encoder must not allocate escaped bytes"))
    with pytest.raises(subject.DiagnosticError,match="report_budget"):
        subject.canonical_bytes({value:0} if as_key else value)


def test_precharged_size_matches_canonical_encoder_and_lexical_strings():
    value={"\n\U0001f600\x7f": ["[{}]\\\"\u00e9",-3,2.5,True,False,None,{}]}
    encoded=subject.canonical_bytes(value)
    assert encoded==json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
    subject._json_preflight(encoded,subject.MAX_REPORT_BYTES)


@pytest.mark.parametrize("change",["graph_reason","graph_detail","zero_calls","multiple_calls"])
def test_resealed_nonunique_graph_proof_joins_are_strict(change,tmp_path,monkeypatch):
    report=_run(tmp_path,monkeypatch).report
    if change=="graph_reason":report["graph"]["reason"]="fabricated_reason"
    elif change=="graph_detail":report["graph"]["detail"]="multiple"
    elif change=="zero_calls":report["g_calls"]=0
    else:report["g_calls"]=2
    report.pop("result_sha256");subject._seal(report)
    with pytest.raises(subject.DiagnosticError):subject.validate_report(report)


@pytest.mark.parametrize("change",["outer_status","outer_reason","outer_field","graph_reason","graph_detail","zero_calls","multiple_calls"])
def test_resealed_budget_refusal_envelope_and_graph_joins_are_exact(change,tmp_path,monkeypatch):
    graph,sources=controlled_graph("budget_refused")
    report=_run(tmp_path,monkeypatch,graph,sources).report
    assert report["status"]=="refused" and report["reason"]=="selection_budget_refused" and report["field"]=="budget"
    if change=="outer_status":report["status"]="incomplete"
    elif change=="outer_reason":report["reason"]="another_refusal"
    elif change=="outer_field":report["field"]="another_field"
    elif change=="graph_reason":report["graph"]["reason"]="another_refusal"
    elif change=="graph_detail":report["graph"]["detail"]="no"
    elif change=="zero_calls":report["g_calls"]=0
    else:report["g_calls"]=2
    report.pop("result_sha256");subject._seal(report)
    with pytest.raises(subject.DiagnosticError):subject.validate_report(report)


CURRENT_EXECUTION_SEAM_PINS = {'src/mc_photogrammetry/pipeline/image_derived_geometry_orchestration.py': 'cbd2d8e2e8e744799628159b4eb307207024435272441d4b90215bb6f10add1e', 'src/mc_photogrammetry/pipeline/image_derived_joint_graph.py': '535c1010aaf41bfad052e27d691038ac44e6bcf92ea98f61c73ac0a3a38bd878', 'src/mc_photogrammetry/pipeline/image_derived_endpoint_selection.py': '6d4a59dd7725fd50a1bc8f85646939907cdbb2a361462fc5f652e8e093b91e1e', 'src/mc_photogrammetry/reconstruction/endpoint_hypothesis.py': '86f8ccfa1665eb523830dce79c8f3593ed4afdb1b1993966c8d82b2036db9b0f', 'src/mc_photogrammetry/reconstruction/plane_graph.py': '916111dbaf0440fd40e752e38664477b8f0a0858cd501d8e2b570598f0d83073'}


def test_exact_current_partial_source_pins_include_imported_matcher():
    assert subject.SOURCE_PINS == CURRENT_EXECUTION_SEAM_PINS
    assert len(subject.SOURCE_PINS) == 5
    for path, digest in CURRENT_EXECUTION_SEAM_PINS.items():
        assert subject._file_hash(subject.REPO_ROOT / path) == digest


@pytest.mark.parametrize("relative_path", tuple(CURRENT_EXECUTION_SEAM_PINS),
                         ids=["orchestration", "joint-graph", "selection", "csp", "plane-graph"])
def test_each_source_pin_mismatch_refuses_before_input_hash_capture_or_delivery(relative_path, tmp_path, monkeypatch):
    image = tmp_path / "ordinary.png"
    image.write_bytes(b"authored synthetic input; never decoded")
    original_hash = subject._file_hash
    seen = []
    def altered_hash(path):
        path = Path(path)
        assert path != image, "source refusal must precede input hash/ingress"
        seen.append(path)
        return "0" * 64 if path == subject.REPO_ROOT / relative_path else original_hash(path)
    monkeypatch.setattr(subject, "_file_hash", altered_hash)
    monkeypatch.setattr(subject, "GraphCapture", lambda: pytest.fail("capture before source pins"))
    def delivery(*args, **kwargs):
        pytest.fail("delivery after mismatched source pin")
    outputs = [tmp_path / "out.mcscan", tmp_path / "out.litematic", tmp_path / "diagnostic.json"]
    with pytest.raises(subject.DiagnosticError) as caught:
        subject.run_once([image], project=outputs[0], destination=outputs[1], diagnostic=outputs[2],
                         name="synthetic", description="synthetic", delivery=delivery)
    assert (caught.value.reason, caught.value.field) == ("source_pin_mismatch", "runtime")
    assert seen == [subject.REPO_ROOT / name for name in CURRENT_EXECUTION_SEAM_PINS]
    assert all(not path.exists() for path in outputs)


def test_source_backed_failure_projection_preserves_full_proof_and_uninstrumented_replay(tmp_path, monkeypatch):
    graph, sources = controlled_graph("unsupported_reading_endpoint")
    original_proof = graph.selection.proof_record()
    original_solver = csp.solve_endpoint_hypotheses
    replays = []
    def replay(*args, **kwargs):
        replays.append((args, kwargs))
        return original_solver(*args, **kwargs)
    monkeypatch.setattr(csp, "solve_endpoint_hypotheses", replay)
    for module, name in [(plane_graph, "match_plane_graph"), (selection, "select_endpoint_hypotheses"),
                         (plane_graph, "_evaluate_assignment")]:
        monkeypatch.setattr(module, name, lambda *a, **k: pytest.fail("diagnostic changed primary inference"))
    primary_calls = []
    def original_callable(**kwargs):
        primary_calls.append(kwargs)
        return graph
    monkeypatch.setattr(subject.orchestration.g_mod, "derive_image_derived_graph", original_callable)
    def delivery(images, **kwargs):
        return subject.orchestration.g_mod.derive_image_derived_graph(**sources)
    result = _run(tmp_path, monkeypatch, delivery=delivery)
    assert result.production_result is graph and result.report["g_calls"] == 1
    assert result.production_error is None and result.diagnostic_error is None
    assert subject.orchestration.g_mod.derive_image_derived_graph is original_callable
    assert len(primary_calls) == 1
    assert all(primary_calls[0][key] is value for key, value in sources.items())
    assert len(replays) == 1  # Existing ImageDerivedGraphResult source-proof validation.
    args, kwargs = replays[0]
    assert args[0] is graph.readings and args[1] is graph.candidate_matches
    assert kwargs.get("endpoint_incidence_sink") is None
    assert kwargs["budget"] == graph.selection.budget
    assert result.report["graph"]["selection"]["proof"] == original_proof
    assert result.report["graph"]["selection"]["result_sha256"] == graph.selection.result_sha256
    assert sum(e["candidate_incidence"] for e in result.report["graph"]["endpoint_incidence"]) == 2 * len(graph.candidate_matches)
    assert graph.selection.proof_record() == original_proof


@pytest.mark.parametrize("kind", ["no_active_factors", "unsupported_reading_endpoint", "budget_refused"])
def test_current_failure_certificate_dispositions_roundtrip_immutable_and_digest_exact(kind):
    graph, _ = controlled_graph(kind)
    record = subject._certificate_record(graph.selection)
    assert type(record["candidate_dispositions"]) is tuple
    direct = subject._certificate_from_record(record)
    transported = json.loads(subject.canonical_bytes(record))
    assert type(transported["candidate_dispositions"]) is list
    rebuilt = subject._certificate_from_record(transported)
    assert type(rebuilt.candidate_dispositions) is tuple
    assert rebuilt.candidate_dispositions == direct.candidate_dispositions == graph.selection.candidate_dispositions
    assert rebuilt.result_sha256 == direct.result_sha256 == graph.selection.result_sha256
    assert selection._payload(rebuilt) == selection._payload(graph.selection)
    assert rebuilt.proof_record() == graph.selection.proof_record()


@pytest.mark.parametrize("change", ["missing", "scalar", "foreign", "nested", "wrong-count", "over-cap", "bool-count"])
def test_current_failure_certificate_malformed_dispositions_refuse_before_constructor(change, monkeypatch):
    graph, _ = controlled_graph("unsupported_reading_endpoint")
    record = json.loads(subject.canonical_bytes(subject._certificate_record(graph.selection)))
    if change == "missing":
        del record["candidate_dispositions"]
    elif change == "scalar":
        record["candidate_dispositions"] = "admitted"
    elif change == "foreign":
        record["candidate_dispositions"][0] = "foreign"
    elif change == "nested":
        record["candidate_dispositions"][0] = ["admitted"]
    elif change == "wrong-count":
        record["candidate_dispositions"].append("admitted")
    elif change == "over-cap":
        record["candidate_count"] = 16385
        record["candidate_dispositions"] = ["admitted"] * 16385
    else:
        record["candidate_count"] = True
    monkeypatch.setattr(selection.EndpointSelectionCertificate, "__post_init__",
                        lambda *args: pytest.fail("malformed transport entered certificate constructor"))
    with pytest.raises(ValueError):
        subject._certificate_from_record(record)


@pytest.mark.parametrize("change", ["disposition", "foreign-proof", "proof-counter"])
def test_current_failure_certificate_tamper_rejected_even_with_resealed_outer_report(change, tmp_path, monkeypatch):
    graph, sources = controlled_graph("unsupported_reading_endpoint")
    result = _run(tmp_path, monkeypatch, graph, sources)
    assert result.diagnostic_error is None
    report = json.loads(subject.canonical_bytes(result.report))
    cert = report["graph"]["selection"]
    if change == "disposition":
        cert["candidate_dispositions"][0] = plane_graph.SAME_AXIS_DISPOSITION_DROPPED
    elif change == "foreign-proof":
        other, _ = controlled_graph("budget_refused")
        cert["proof"] = other.selection.proof_record()
    else:
        cert["proof"]["counters"]["eligible_assignments"] += 1
    report.pop("result_sha256")
    subject._seal(report)
    with pytest.raises(subject.DiagnosticError, match="malformed_report: graph_proof"):
        subject.validate_report(report)


def _seed_finish_fields(count, *, nodes=1):
    """Authored scalar hook payload, not a substitute for production predicates."""
    return dict(node_count=nodes, side_match_count=count,
        normal_score_pass_count=count, margin_pass_count=count, tangent_support_pass_count=count,
        after_side_count=count, after_score_count=count, after_margin_count=count, after_support_count=count,
        max_side_normal_score=None if not count else 0.5,
        max_side_margin=None if not count else 0.4,
        max_side_tangent_support=None if not count else 20.0)


def _emit_attrition(observer, readings, *, rejected=(), corruption=False, emit_seeds=True):
    """Authored scalar hook events; no image/pipeline call or numeric evaluator."""
    frames = tuple(sorted({r.frame for r in readings}))
    for frame in frames:
        frame_rows = sorted((r for r in readings if r.frame == frame), key=lambda r: (r.axis, r.side))
        for axis in range(3):
            for side in (-1, 1):
                if not emit_seeds:
                    continue
                identity = (frame, axis, side)
                count = sum(r.axis == axis and r.side == side for r in frame_rows)
                observer.expect("seed", identity)
                observer.finish("seed", identity, "available" if count else "absent",
                                axis=axis, side=side, eligible_seed_count=count,
                                **_seed_finish_fields(count, nodes=max(1, len(frame_rows))))
        for plane, reading in enumerate(frame_rows):
            identity = (frame, plane)
            observer.expect("proposal", identity)
            observer.finish("proposal", identity, "returned", axis=int(reading.axis), side=int(reading.side),
                            returned_ordinal=plane, discarded_coordinates=[12345, 67890])
            observer.bind(("proposal_return", frame, plane), identity)
            observer.expect("reading", identity)
            observer.finish("reading", identity, "accepted", axis=(reading.axis + 1) % 3 if corruption else int(reading.axis),
                            side=float(reading.side), plane_index=plane, discarded_path="private/example.png")
            observer.bind(("reading", id(reading)), identity)
        for offset, reason in enumerate(rejected, len(frame_rows)):
            identity = (frame, offset)
            observer.expect("proposal", identity)
            observer.finish("proposal", identity, reason, axis=0, side=1)


def _attrition_fixture(*, axes=(1,), rejected=(), corruption=False, observer=None):
    observer = subject.ReadingAttritionObserver() if observer is None else observer
    readings = tuple(SimpleNamespace(frame=frame, axis=axis, side=1) for frame in ("image-0000", "image-0001") for axis in axes)
    _emit_attrition(observer, readings, rejected=rejected, corruption=corruption)
    capture = subject.GraphCapture(observer)
    capture.graph = SimpleNamespace(frame_ids=("image-0000", "image-0001"))
    capture.arguments = ((), dict(lattice_result=SimpleNamespace(readings=readings)))
    return observer, capture, readings


def test_reading_attrition_complete_axis1_only_missing_axis0_and_scalar_redaction():
    observer, capture, readings = _attrition_fixture(rejected=("deduplicated", "diversity_cap", "area_rejected"))
    record = subject._reading_attrition_record(observer, capture)
    assert record["status"] == "complete" and all(record["checks"].values())
    assert all(row["proposal_axis"] == row["reading_axis"] == 1 for row in record["lineage"])
    assert all(row["axis"] == 1 for row in record["returned_readings"])
    zero_seeds = [row for row in record["stages"]["seed"]["terminal"] if row["axis"] == 0]
    assert len(zero_seeds) == 4 and all(row["eligible_seed_count"] == 0 for row in zero_seeds)
    assert {r["disposition"] for r in record["counts"] if r["stage"] == "proposal"} == {"returned", "deduplicated", "diversity_cap", "area_rejected"}
    encoded = subject.canonical_bytes(record)
    assert len(encoded) < subject.ATTRITION_BYTES
    assert all(token not in encoded for token in (b"12345", b"67890", b"private", b"example.png", b"coordinates", b"mask", b"pixels"))
    assert tuple(capture.arguments[1]["lattice_result"].readings) == readings


@pytest.mark.parametrize("quadrant,expected", [
    (0, ((0, 1), (1, 1), (2, 1))), (1, ((2, 1), (1, 1), (0, -1))),
    (2, ((0, -1), (1, 1), (2, -1))), (3, ((2, -1), (1, 1), (0, 1))),
])
def test_reading_attrition_axis_lineage_preserves_all_quadrant_mappings(quadrant, expected):
    observer, capture, _ = _attrition_fixture(axes=(0, 1, 2))
    record = subject._reading_attrition_record(observer, capture)
    assert record["status"] == "complete"
    assert tuple(plane_graph.axis_transform(axis, quadrant) for axis in range(3)) == expected
    assert all(plane_graph.axis_transform(row["proposal_axis"], quadrant) == plane_graph.axis_transform(row["reading_axis"], quadrant) for row in record["lineage"])
    assert plane_graph.axis_transform(1, quadrant) == (1, 1)


@pytest.mark.parametrize("reason", ["periodicity", "coverage", "reading_cap", "per_axis_side_cap", "window_incidence"])
def test_reading_attrition_reader_refusal_and_caps_are_distinct_from_proposal_refusal(reason):
    observer, capture, _ = _attrition_fixture(axes=(1,), rejected=("deduplicated",))
    # Authored returned proposal that reader rejects; it creates no output reading.
    identity = ("image-0000", 2)
    observer.expect("proposal", identity)
    observer.finish("proposal", identity, "returned", axis=0, side=1)
    observer.bind(("proposal_return", *identity), ("image-0000", 1))
    observer.expect("reading", ("image-0000", 1))
    observer.finish("reading", ("image-0000", 1), reason, axis=0, side=None if reason == "window_incidence" else 1)
    record = subject._reading_attrition_record(observer, capture)
    assert record["status"] == "complete"
    joined = next(row for row in record["lineage"] if row["generated"] == 2)
    assert joined["proposal_axis"] == joined["reading_axis"] == 0 and joined["disposition"] == reason and joined["plane_index"] is None
    assert len(record["returned_readings"]) == 2


def test_reading_attrition_axis_corruption_is_incomplete_not_production_refusal():
    observer, capture, originals = _attrition_fixture(corruption=True)
    record = subject._reading_attrition_record(observer, capture)
    assert record["status"] == "incomplete" and not record["checks"]["axis_lineage"]
    assert not record["checks"]["accepted_reading_join"]
    assert all(r.axis == 1 for r in originals)


@pytest.mark.parametrize("failure", ["omitted", "duplicate", "missing", "foreign", "item-cap", "byte-cap"])
def test_reading_attrition_fail_closed_accounting_and_overflow(failure):
    observer, capture, _ = _attrition_fixture()
    if failure == "omitted":
        observer.omitted += 1
    elif failure == "duplicate":
        observer.finish("reading", ("image-0000", 0), "accepted", axis=1, side=1, plane_index=0)
    elif failure == "missing":
        observer.expect("reading", ("image-0000", 99))
    elif failure == "foreign":
        observer.finish("reading", ("image-0000", 99), "periodicity", axis=1)
    elif failure == "item-cap":
        observer, capture, _ = _attrition_fixture(observer=subject.ReadingAttritionObserver(max_items=1))
    else:
        observer, capture, _ = _attrition_fixture(observer=subject.ReadingAttritionObserver(max_bytes=1))
    record = subject._reading_attrition_record(observer, capture)
    assert record is None or record["status"] == "incomplete"
    assert observer.incomplete or observer.omitted or failure in ("missing", "foreign")


@pytest.mark.parametrize("change", ["axis", "handoff", "counts", "path", "fields", "bound"])
def test_reading_attrition_serialized_tamper_and_schema_bounds(change):
    observer, capture, _ = _attrition_fixture()
    record = json.loads(subject.canonical_bytes(subject._reading_attrition_record(observer, capture)))
    if change == "axis": record["lineage"][0]["reading_axis"] = 0
    elif change == "handoff": record["handoffs"][0]["returned"][1] = 9
    elif change == "counts": record["counts"][0]["count"] += 1
    elif change == "path": record["stages"]["seed"]["terminal"][0]["disposition"] = "private/secret.png"
    elif change == "fields": record["stages"]["seed"]["terminal"][0]["pixels"] = [1, 2]
    else: record["stages"]["reading"]["terminal"] *= subject.ATTRITION_ITEMS + 1
    try:
        checks, lineage, counts, status = subject._attrition_account(record)
    except ValueError:
        return
    assert (record["checks"], record["lineage"], record["counts"], record["status"]) != (checks, lineage, counts, status)


def test_reading_attrition_context_same_result_one_call_and_exception_restoration(monkeypatch):
    graph, sources = controlled_graph("unsupported_reading_endpoint")
    marker = object()
    calls = []
    def original(**kwargs):
        calls.append(kwargs)
        assert all(kwargs[k] is value for k, value in sources.items())
        return graph
    monkeypatch.setattr(subject.orchestration.g_mod, "derive_image_derived_graph", original)
    measure_args, measure_kwargs, measure_result, measured = object(), object(), object(), []
    def original_measure(value, *, keyword):
        assert value is measure_args and keyword is measure_kwargs
        measured.append((value, keyword))
        observer = subject.region_growth._attrition_state.get()
        if isinstance(observer, subject.ReadingAttritionObserver):
            _emit_attrition(observer, sources["lattice_result"].readings)
        return measure_result
    monkeypatch.setattr(subject.lattice_measure, "_measure_frame", original_measure)
    def delivery():
        before = subject.region_growth._attrition_state.get()
        assert subject.lattice_measure._measure_frame(measure_args, keyword=measure_kwargs) is measure_result
        assert subject.region_growth._attrition_state.get() is before
        return subject.orchestration.g_mod.derive_image_derived_graph(**sources)
    assert subject.GraphCapture().invoke(delivery) is graph
    observer = subject.ReadingAttritionObserver()
    capture = subject.GraphCapture(observer)
    with subject.region_growth._scalar_attrition_context(marker, frame="outer"):
        assert capture.invoke(delivery) is graph
        assert subject.region_growth._attrition_state.get() is marker
        assert subject.region_growth._attrition_frame.get() == "outer"
        exact_error = RuntimeError("must propagate identical exception")
        def raising_measure(*args, **kwargs):
            assert args == (measure_args,) and kwargs["keyword"] is measure_kwargs
            assert subject.region_growth._attrition_state.get() is observer
            with subject.region_growth._scalar_attrition_context(frame="inner"):
                raise exact_error
        monkeypatch.setattr(subject.lattice_measure, "_measure_frame", raising_measure)
        def failed():
            return subject.lattice_measure._measure_frame(measure_args, keyword=measure_kwargs)
        with pytest.raises(RuntimeError) as caught: capture.invoke(failed)
        assert caught.value is exact_error and subject.region_growth._attrition_state.get() is marker
        assert subject.region_growth._attrition_frame.get() == "outer"
        assert subject.lattice_measure._measure_frame is raising_measure
    assert len(calls) == 2 and subject.orchestration.g_mod.derive_image_derived_graph is original
    assert measured == [(measure_args, measure_kwargs), (measure_args, measure_kwargs)]
    assert subject.region_growth._attrition_state.get() is None


def test_reading_attrition_report_digest_and_incomplete_preserve_graph(monkeypatch, tmp_path):
    graph, sources = controlled_graph("unsupported_reading_endpoint")
    off = _run(tmp_path, monkeypatch, graph, sources)
    capture = capture_for(graph, sources)
    observer = subject.ReadingAttritionObserver()
    _emit_attrition(observer, sources["lattice_result"].readings)
    capture.reading_attrition = observer
    on = subject.build_report(capture, deepcopy(off.report["run"]))
    assert on["status"] == "complete" and on["reading_attrition"]["status"] == "complete"
    assert on["graph"] == off.report["graph"] and graph.result_sha256 == off.production_result.result_sha256
    assert on["result_sha256"] != off.report["result_sha256"]
    assert "every admitted candidate row unchanged" in on["notes"][1]
    corrupt = deepcopy(on); corrupt["reading_attrition"]["lineage"][0]["proposal_axis"] = 2
    with pytest.raises(subject.DiagnosticError): subject.validate_report(corrupt)
    corrupt.pop("result_sha256"); subject._seal(corrupt)
    with pytest.raises(subject.DiagnosticError, match="reading_attrition"): subject.validate_report(corrupt)
    observer.incomplete = True
    incomplete = subject.build_report(capture, deepcopy(off.report["run"]))
    assert incomplete["status"] == "incomplete" and incomplete["reason"] == "reading_attrition_incomplete"
    assert incomplete["graph"] == off.report["graph"] and graph is off.production_result


def test_all_admitted_candidate_rows_and_dispositions_remain_unpruned(monkeypatch):
    from test_endpoint_hypothesis import make_match
    graph, sources = controlled_graph("unsupported_reading_endpoint")
    reading_index, _, _ = plane_graph._canonical_reading_index(graph.readings)
    left_plane = next(i for (f, i), r in reading_index.items() if f == "a" and r.axis == 0)
    right_plane = next(i for (f, i), r in reading_index.items() if f == "b" and r.axis == 2)
    mismatch = make_match("a", "b", 0, 2, 1, left_plane, right_plane)
    rows = graph.candidate_matches + (mismatch,)
    cert = selection.select_endpoint_hypotheses(readings=graph.readings, signatures=graph.signatures,
        candidate_matches=rows, frame_ids=graph.frame_ids, quadrant_constraints=(("a", "b", 0),),
        budget=graph.selection.budget,
        candidate_dispositions=graph.selection.candidate_dispositions + ("same_axis_mismatch",),
        session_sha256="a"*64, lattice_sha256="b"*64, signatures_sha256="c"*64)
    assert cert.candidate_count == len(rows) and cert.candidate_dispositions[-1] == "same_axis_mismatch"
    assert cert.candidate_matches_sha256 == selection.candidate_digest(rows)
    fields = {field.name: getattr(graph, field.name) for field in dataclasses.fields(graph)
              if field.name != "result_sha256"}
    fields.update(reason=cert.reason, detail=cert.status, candidate_matches=rows, selection=cert)
    graph = graph_mod.ImageDerivedGraphResult(**fields)
    capture = capture_for(graph, sources)
    record = subject._graph_record(capture)
    assert graph.candidate_matches is rows and rows[-1] is mismatch
    assert capture.graph.candidate_matches is rows and capture.graph.candidate_matches[-1] is mismatch
    assert record["selection"]["candidate_matches_sha256"] == selection.candidate_digest(rows)
    assert record["selection"]["candidate_count"] == len(rows)
    assert record["selection"]["candidate_dispositions"][-1] == "same_axis_mismatch"
    assert sum(r["candidate_incidence"] for r in record["endpoint_incidence"]) == 2 * len(rows)


def test_reading_attrition_observer_failure_does_not_change_inference(monkeypatch):
    observer, capture, readings = _attrition_fixture()
    sentinel = object()
    exact_args = object()
    calls = []
    def broken_reservation(size):
        raise RuntimeError("observer failure; must never escape inference")
    monkeypatch.setattr(observer, "_reserve", broken_reservation)
    def measured(value):
        calls.append(value)
        current = subject.region_growth._attrition_state.get()
        current.expect("proposal", ("image-0000", 99))
        current.finish("proposal", ("image-0000", 99), "area_rejected", axis=0, side=1)
        return sentinel
    monkeypatch.setattr(subject.lattice_measure, "_measure_frame", measured)
    def delivery(value):
        assert subject.region_growth._attrition_state.get() is None
        return subject.lattice_measure._measure_frame(value)
    assert capture.invoke(delivery, exact_args) is sentinel
    assert calls == [exact_args] and observer.incomplete
    assert subject.region_growth._attrition_state.get() is None
    assert readings[0].axis == 1


def test_reading_attrition_downstream_signature_matcher_and_replay_cannot_charge_budget(monkeypatch):
    observer = subject.ReadingAttritionObserver()
    sentinel = object()
    def measured(frame, calibration):
        assert subject.region_growth._attrition_state.get() is observer
        observer.expect("seed", ("image-0000", 0, 1))
        observer.finish("seed", ("image-0000", 0, 1), "absent", axis=0, side=1,
                        eligible_seed_count=0, **_seed_finish_fields(0))
        return sentinel
    monkeypatch.setattr(subject.lattice_measure, "_measure_frame", measured)
    downstream = []
    def downstream_hook(stage):
        downstream.append(stage)
        assert subject.region_growth._attrition_state.get() is None
        before = observer.bytes, observer.omitted, deepcopy(observer.scalar_rows)
        # Source hooks do nothing without an observer; no injection into inference.
        current = subject.region_growth._attrition_state.get()
        if current is not None:
            current.expect(stage, ("image-0000", 0))
            current.finish(stage, ("image-0000", 0), "available")
        assert before == (observer.bytes, observer.omitted, observer.scalar_rows)
    def delivery():
        assert subject.lattice_measure._measure_frame(object(), object()) is sentinel
        for stage in ("signature", "pair", "reading"):
            downstream_hook(stage)
        return sentinel
    assert subject.GraphCapture(observer).invoke(delivery) is sentinel
    assert downstream == ["signature", "pair", "reading"]
    assert set(observer.stages) == {"seed"}
    assert subject.lattice_measure._measure_frame is measured


@pytest.mark.parametrize("stage,wrong", [("seed", "returned"), ("proposal", "accepted"), ("reading", "available")])
def test_reading_attrition_unknown_or_wrong_stage_terminal_is_incomplete(stage, wrong):
    observer, capture, _ = _attrition_fixture()
    identity = ("image-0000", 2, -1) if stage == "seed" else ("image-0000", 90)
    observer.expect(stage, identity)
    observer.finish(stage, identity, wrong, axis=2, side=-1)
    assert observer.incomplete
    record = subject._reading_attrition_record(observer, capture)
    assert record is None or record["status"] == "incomplete"
    clean, capture, _ = _attrition_fixture()
    record = subject._reading_attrition_record(clean, capture)
    record["stages"][stage]["terminal"][0]["disposition"] = wrong
    with pytest.raises(ValueError, match="disposition"):
        subject._attrition_account(record)


def test_reading_attrition_full_envelope_budget_charges_lineage_counts_checks():
    observer, capture, _ = _attrition_fixture(axes=(0, 1, 2))
    record = subject._reading_attrition_record(observer, capture)
    full_bytes = len(subject.canonical_bytes(record))
    raw_bytes = len(subject.canonical_bytes({k: record[k] for k in ("frames", "stages", "handoffs", "returned_readings")}))
    assert raw_bytes < full_bytes - 128
    record["limits"]["bytes"] = full_bytes - 128
    with pytest.raises(ValueError, match="full envelope byte cap"):
        subject._attrition_account(record)


def test_reading_attrition_actual_measure_proposal_seed_and_reader_hooks_synthetic(monkeypatch):
    import numpy as np
    from test_region_growth import _synthetic_room, _calibration
    from mc_photogrammetry.reconstruction import lattice_planes
    image, _usable, segments, _obstacle = _synthetic_room()
    calibration = _calibration()
    frame = SimpleNamespace(frame_id="image-0000", image=image)
    observer = subject.ReadingAttritionObserver()
    monkeypatch.setattr(subject.region_growth, "detect_segments", lambda grey: segments)
    monkeypatch.setattr(lattice_planes, "detect_segments", lambda grey: segments)
    # Keep this a deterministic hook integration control: the numerical period
    # seam abstains, while the real proposal/seed/reader terminal hooks execute.
    period_calls = []
    def period_abstains(*args, **kwargs):
        assert subject.region_growth._attrition_state.get() is observer
        period_calls.append(kwargs)
        return None, None, None, None, None
    monkeypatch.setattr(lattice_planes, "_bounded_measure", period_abstains)
    # Restrict masking to this authored synthetic unit fixture, not a product setting.
    monkeypatch.setattr(subject.lattice_measure, "create_gui_mask", lambda h, w: np.zeros((h, w), dtype=bool))
    original_measure = subject.lattice_measure._measure_frame
    original_proposals = subject.region_growth.propose_image_planes_v2
    original_reader = lattice_planes.frame_plane_readings_from_proposals
    counts = dict(proposal=0, reader=0)
    def proposals(*args, **kwargs):
        counts["proposal"] += 1
        return original_proposals(*args, **kwargs)
    def reader(*args, **kwargs):
        counts["reader"] += 1
        return original_reader(*args, **kwargs)
    monkeypatch.setattr(subject.region_growth, "propose_image_planes_v2", proposals)
    monkeypatch.setattr(lattice_planes, "frame_plane_readings_from_proposals", reader)
    capture = subject.GraphCapture(observer)
    result = capture.invoke(lambda: subject.lattice_measure._measure_frame(frame, calibration))
    assert counts == dict(proposal=1, reader=1)
    assert subject.lattice_measure._measure_frame is original_measure
    assert subject.region_growth._attrition_state.get() is None
    assert subject.region_growth._attrition_frame.get() == ""
    assert len(observer.stages["seed"]["expected"]) == 6
    assert observer.stages["proposal"]["expected"]
    assert observer.stages["reading"]["expected"]
    assert period_calls
    assert any(row["disposition"] == "returned" for row in observer.scalar_rows["proposal"].values())
    assert set(observer.stages["reading"]["expected"]) == set(observer.stages["reading"]["terminal"])
    readings, abstentions = result
    capture.graph = SimpleNamespace(frame_ids=(frame.frame_id,))
    capture.arguments = ((), dict(lattice_result=SimpleNamespace(readings=readings)))
    record = subject._reading_attrition_record(observer, capture)
    assert record is not None and record["status"] == "complete"
    assert record["checks"]["axis_lineage"] and record["checks"]["accepted_reading_join"]
    assert record["schema"] == subject.ATTRITION_SCHEMA == "g7-reading-attrition-v2"
    assert record["checks"]["seed_node_count_consistency"]
    source_seed_rows = {tuple(row["identity"]): row for row in observer.record()["stages"]["seed"]["examples"]}
    for row in record["stages"]["seed"]["terminal"]:
        assert row["schema"] == subject.ATTRITION_ROW_SCHEMA
        original = source_seed_rows[tuple(row["identity"])]
        assert row["seed_predicates"] == {name: original[name]
            for name in subject.SEED_COUNT_FIELDS + subject.SEED_MAX_FIELDS}
        assert row["eligible_seed_count"] == original["eligible_seed_count"]
    assert all(row["disposition"] in subject.ATTRITION_DISPOSITIONS["reading"] for row in record["stages"]["reading"]["terminal"])
    assert len(record["returned_readings"]) == len(readings)
    assert isinstance(abstentions, tuple)
    encoded = subject.canonical_bytes(record)
    assert all(word not in encoded for word in (b'"polygon":', b'"pixels":', b'"coordinates":', b'"camera_pose":'))


def _seed_predicate_hook(graph, *, axis=1, side=1, observer=None):
    """Execute the original hook on authored atom arrays; no image/product ingress."""
    rg = subject.region_growth
    observer = subject.ReadingAttritionObserver() if observer is None else observer
    assert not hasattr(observer, "seed_predicates")  # Never request the array sink.
    with rg._scalar_attrition_context(observer, frame="image-0000"):
        result = rg._seed_and_grow(graph, axis, side, rg.DEFAULT_REGION_GROWTH_CONFIG)
    assert rg._attrition_state.get() is None and rg._attrition_frame.get() == ""
    rows = list(observer.scalar_rows["seed"].values())
    return result, observer, rows[0] if rows else None


def test_seed_predicate_hook_exact_scalar_projection_and_output_neutral():
    from test_region_growth import _seed_predicate_graph
    graph = _seed_predicate_graph()
    original = subject.region_growth._seed_and_grow(graph, 1, 1,
        subject.region_growth.DEFAULT_REGION_GROWTH_CONFIG)
    result, observer, row = _seed_predicate_hook(graph)
    assert result == original == [[3]] and not observer.incomplete
    assert row["schema"] == subject.ATTRITION_ROW_SCHEMA
    assert row["eligible_seed_count"] == 1 and row["disposition"] == "available"
    predicates = row["seed_predicates"]
    assert {name: predicates[name] for name in subject.SEED_COUNT_FIELDS} == dict(
        node_count=5, side_match_count=4, normal_score_pass_count=4, margin_pass_count=4,
        tangent_support_pass_count=4, after_side_count=4, after_score_count=3,
        after_margin_count=2, after_support_count=1)
    assert predicates["max_side_normal_score"] == pytest.approx(0.5)
    assert predicates["max_side_margin"] == pytest.approx(0.4)
    assert predicates["max_side_tangent_support"] == pytest.approx(20)
    original_hook_row = observer.record()["stages"]["seed"]["examples"][0]
    assert predicates == {name: original_hook_row[name] for name in subject.SEED_COUNT_FIELDS + subject.SEED_MAX_FIELDS}
    assert all(type(value) in (int, float, type(None)) for value in predicates.values())
    assert set(predicates) == set(subject.SEED_COUNT_FIELDS + subject.SEED_MAX_FIELDS)


@pytest.mark.parametrize("permutation", [(0, 1, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)])
def test_seed_predicate_axis_permutation_relabels_source_rows_equivariantly(permutation):
    from test_region_growth import _seed_predicate_graph
    graph = _seed_predicate_graph()
    permuted = dataclasses.replace(graph, supports=graph.supports[:, permutation],
        plane_scores=graph.plane_scores[:, permutation], sides=graph.sides[:, permutation],
        axis_line_dirs=graph.axis_line_dirs[:, permutation])
    for axis in range(3):
        for side in (-1, 1):
            output, observer, row = _seed_predicate_hook(graph, axis=axis, side=side)
            relabeled_output, relabeled_observer, relabeled = _seed_predicate_hook(
                permuted, axis=permutation.index(axis), side=side)
            assert not observer.incomplete and not relabeled_observer.incomplete
            assert output == relabeled_output
            assert relabeled["identity"] == ["image-0000", permutation.index(axis), side]
            assert relabeled["axis"] == permutation.index(axis) and relabeled["side"] == side
            assert relabeled["seed_predicates"] == row["seed_predicates"]
            assert relabeled["eligible_seed_count"] == row["eligible_seed_count"]


def test_seed_predicate_side_reflection_swaps_rows_preserves_side_union():
    from test_region_growth import _seed_predicate_graph
    graph = _seed_predicate_graph()
    reflected = dataclasses.replace(graph, sides=-graph.sides)
    for axis in range(3):
        union, reflected_union = [], []
        for side in (-1, 1):
            output, _, row = _seed_predicate_hook(graph, axis=axis, side=side)
            swapped, _, reflected_row = _seed_predicate_hook(reflected, axis=axis, side=-side)
            assert output == swapped and row["seed_predicates"] == reflected_row["seed_predicates"]
            union.append(row["eligible_seed_count"])
            reflected_union.append(reflected_row["eligible_seed_count"])
        assert sum(union) == sum(reflected_union)


@pytest.mark.parametrize("predicate", ["score", "margin", "support"])
@pytest.mark.parametrize("at_boundary", [True, False], ids=["equal", "below"])
def test_seed_predicate_actual_default_threshold_equality_and_just_below(predicate, at_boundary):
    import numpy as np
    from test_region_growth import _seed_predicate_graph
    original = _seed_predicate_graph()
    config = subject.region_growth.DEFAULT_REGION_GROWTH_CONFIG
    scores = np.zeros((5, 3), dtype=np.float64)
    supports = np.zeros((5, 3), dtype=np.float64)
    sides = np.ones((5, 3), dtype=np.int8)
    threshold = dict(score=config.min_seed_score, margin=config.min_seed_margin,
                     support=config.min_seed_support)[predicate]
    value = threshold if at_boundary else np.nextafter(threshold, -np.inf)
    scores[0, 0] = value if predicate in ("score", "margin") else 0.5
    supports[0, 1] = value if predicate == "support" else config.min_seed_support
    graph = dataclasses.replace(original, plane_scores=scores, supports=supports, sides=sides)
    _, observer, row = _seed_predicate_hook(graph, axis=0)
    field = dict(score="normal_score_pass_count", margin="margin_pass_count",
                 support="tangent_support_pass_count")[predicate]
    assert not observer.incomplete and row["seed_predicates"][field] == int(at_boundary)
    if predicate != "margin":
        assert row["eligible_seed_count"] == int(at_boundary)
    else:
        # Margin equality is tested directly at best-other==threshold. This
        # low-score atom does not fabricate a jointly eligible seed.
        assert row["seed_predicates"]["after_score_count"] == row["eligible_seed_count"] == 0


def test_seed_predicate_raw_overlapping_pass_sets_are_not_sequential_or_monotone():
    import numpy as np
    from test_region_growth import _seed_predicate_graph
    graph = _seed_predicate_graph()
    graph = dataclasses.replace(graph, supports=np.full((5, 3), 10.0, dtype=np.float32))
    _, observer, row = _seed_predicate_hook(graph)
    values = row["seed_predicates"]
    assert not observer.incomplete
    assert tuple(values[name] for name in ("normal_score_pass_count", "margin_pass_count", "tangent_support_pass_count")) == (4, 4, 5)
    assert tuple(values[name] for name in ("after_side_count", "after_score_count", "after_margin_count", "after_support_count")) == (4, 3, 2, 2)
    assert row["eligible_seed_count"] == 2
    subject._attrition_row("seed", row)  # A larger raw support count is valid.


def test_seed_predicate_zero_side_match_has_null_maxima_and_zero_eligible():
    import numpy as np
    from test_region_growth import _seed_predicate_graph
    graph = _seed_predicate_graph()
    graph = dataclasses.replace(graph, sides=np.ones_like(graph.sides))
    output, observer, row = _seed_predicate_hook(graph, side=-1)
    values = row["seed_predicates"]
    assert output == [] and not observer.incomplete and row["disposition"] == "absent"
    assert row["eligible_seed_count"] == values["side_match_count"] == values["after_support_count"] == 0
    assert all(values[name] is None for name in subject.SEED_MAX_FIELDS)
    assert values["normal_score_pass_count"] > 0  # Independent of side membership.


@pytest.mark.parametrize("change", ["bool", "negative", "over-nodes", "chain", "side-join", "eligible",
    "disposition", "infinite", "bool-max", "extra", "missing", "null-count", "old-row-schema"])
def test_seed_predicate_malformed_or_tampered_rows_fail_closed(change):
    from test_region_growth import _seed_predicate_graph
    _, _, row = _seed_predicate_hook(_seed_predicate_graph())
    row = deepcopy(row); values = row["seed_predicates"]
    if change == "bool": values["node_count"] = True
    elif change == "negative": values["margin_pass_count"] = -1
    elif change == "over-nodes": values["normal_score_pass_count"] = values["node_count"] + 1
    elif change == "chain": values["after_score_count"] = values["after_side_count"] + 1
    elif change == "side-join": values["side_match_count"] -= 1
    elif change == "eligible": row["eligible_seed_count"] = 0
    elif change == "disposition": row["disposition"] = "absent"
    elif change == "infinite": values["max_side_margin"] = float("inf")
    elif change == "bool-max": values["max_side_margin"] = True
    elif change == "extra": values["pixels"] = [1, 2]
    elif change == "missing": values.pop("after_margin_count")
    elif change == "null-count": values["tangent_support_pass_count"] = None
    else: row["schema"] = "g7-reading-attrition-row-v1"
    with pytest.raises(ValueError): subject._attrition_row("seed", row)


def test_seed_predicate_not_reached_is_explicitly_unmeasured_no_fabricated_counts():
    observer = subject.ReadingAttritionObserver()
    identity = ("image-0000", 0, -1)
    observer.expect("seed", identity)
    observer.finish("seed", identity, "not_reached", axis=0, side=-1,
                    eligible_seed_count=None, reason="proposal_abstained")
    row = next(iter(observer.scalar_rows["seed"].values()))
    assert not observer.incomplete and row["seed_predicates"] is None and row["eligible_seed_count"] is None
    for eligible, predicates in ((0, None), (None, _seed_finish_fields(0))):
        bad = dict(row, eligible_seed_count=eligible, seed_predicates=predicates)
        with pytest.raises(ValueError): subject._attrition_row("seed", bad)
    bad_observer = subject.ReadingAttritionObserver()
    bad_observer.expect("seed", identity)
    bad_observer.finish("seed", identity, "not_reached", axis=0, side=-1,
                        eligible_seed_count=None, node_count=0)
    assert bad_observer.incomplete and not bad_observer.scalar_rows["seed"]


def test_seed_predicate_frame_node_count_mismatch_is_incomplete():
    observer, capture, _ = _attrition_fixture(axes=(0, 1, 2))
    row = next(iter(observer.scalar_rows["seed"].values()))
    row["seed_predicates"]["node_count"] += 1
    record = subject._reading_attrition_record(observer, capture)
    assert record["status"] == "incomplete" and not record["checks"]["seed_node_count_consistency"]


def test_seed_predicate_actual_measure_early_return_backfills_unmeasured_rows(monkeypatch):
    import numpy as np
    frame = SimpleNamespace(frame_id="image-0000", image=np.zeros((20, 20, 3), dtype=np.uint8))
    early = SimpleNamespace(status="unavailable", reason="no_eligible_atoms", proposals=())
    calls = []
    def proposals(image, usable, calibration, **kwargs):
        calls.append((image, usable, calibration, kwargs))
        return early
    monkeypatch.setattr(subject.region_growth, "propose_image_planes_v2", proposals)
    monkeypatch.setattr(subject.lattice_measure, "create_gui_mask", lambda h, w: np.zeros((h, w), dtype=bool))
    original_measure = subject.lattice_measure._measure_frame
    observer = subject.ReadingAttritionObserver()
    capture = subject.GraphCapture(observer)
    readings, abstentions = capture.invoke(lambda: subject.lattice_measure._measure_frame(frame, None))
    assert len(calls) == 1 and readings == () and len(abstentions) == 1
    assert subject.lattice_measure._measure_frame is original_measure
    assert subject.region_growth._attrition_state.get() is None
    rows = list(observer.scalar_rows["seed"].values())
    assert len(rows) == 6 and not observer.incomplete
    assert all(row["disposition"] == "not_reached" and row["eligible_seed_count"] is None
               and row["seed_predicates"] is None for row in rows)


def test_seed_predicate_report_digest_and_resealed_invalid_chain_refuses(monkeypatch, tmp_path):
    graph, sources = controlled_graph("unsupported_reading_endpoint")
    off = _run(tmp_path, monkeypatch, graph, sources)
    capture = capture_for(graph, sources)
    observer = subject.ReadingAttritionObserver()
    _emit_attrition(observer, sources["lattice_result"].readings)
    capture.reading_attrition = observer
    on = subject.build_report(capture, deepcopy(off.report["run"]))
    corrupt = deepcopy(on)
    row = next(row for row in corrupt["reading_attrition"]["stages"]["seed"]["terminal"]
               if row["disposition"] == "available")
    row["seed_predicates"]["after_support_count"] = 0
    with pytest.raises(subject.DiagnosticError): subject.validate_report(corrupt)
    corrupt.pop("result_sha256"); subject._seal(corrupt)
    assert corrupt["result_sha256"] != on["result_sha256"]
    with pytest.raises(subject.DiagnosticError, match="reading_attrition"):
        subject.validate_report(corrupt)
    assert on["graph"] == off.report["graph"] and graph is off.production_result


@pytest.mark.parametrize("limit", ["items", "bytes"])
def test_seed_predicate_observer_cap_failure_preserves_original_growth(limit):
    from test_region_growth import _seed_predicate_graph
    graph = _seed_predicate_graph()
    original = subject.region_growth._seed_and_grow(graph, 1, 1,
        subject.region_growth.DEFAULT_REGION_GROWTH_CONFIG)
    observer = subject.ReadingAttritionObserver(**{"max_" + limit: 1})
    output, observer, row = _seed_predicate_hook(graph, observer=observer)
    assert output == original == [[3]] and row is None
    assert observer.incomplete or observer.omitted


def _support_graph(components, *, sides=None):
    """Five authored atoms; real score function, no detector/image/product call."""
    import numpy as np
    from test_region_growth import _seed_predicate_graph
    graph = _seed_predicate_graph()
    supports = np.asarray(components, dtype=np.float64)
    if supports.shape == (3,):
        supports = np.tile(supports, (5, 1))
    assert supports.shape == (5, 3)
    return dataclasses.replace(graph, supports=supports,
        plane_scores=subject.region_growth._normal_scores(supports),
        sides=np.ones((5, 3), dtype=np.int8) if sides is None else sides)


def _observe_support(graph, *, frame="image-0000", observer=None):
    rg = subject.region_growth
    observer = subject.ReadingAttritionObserver() if observer is None else observer
    observer.support_begin(frame)
    with rg._scalar_attrition_context(observer, frame=frame):
        observer.support_graph(frame, (graph, {}))
        outputs = {(axis, side): rg._seed_and_grow(graph, axis, side, rg.DEFAULT_REGION_GROWTH_CONFIG)
                   for axis in range(3) for side in (-1, 1)}
    capture = subject.GraphCapture(observer)
    capture.graph = SimpleNamespace(frame_ids=(frame,))
    capture.arguments = ((), dict(lattice_result=SimpleNamespace(readings=())))
    attrition = subject._reading_attrition_record(observer, capture)
    support = subject._support_score_record(observer, capture, attrition)
    return observer, capture, attrition, support, outputs


@pytest.mark.parametrize("components,axis,score_pass,pattern", [
    ((40, 0, 40), 1, True, 5),
    ((0, 40, 40), 0, True, 6),
    ((0, 1, 79), 0, False, 6),
    ((240, 40, 40), 0, False, 7),
    ((0, 1, 1), 0, False, 6),
], ids=["missing-family1", "balanced", "imbalanced", "normal-contamination", "low-reliability"])
def test_support_score_actual_producer_factors_and_seed_loss(components, axis, score_pass, pattern):
    import math
    graph = _support_graph(components)
    observer, _capture, attrition, record, outputs = _observe_support(graph)
    assert attrition["status"] == record["status"] == "complete"
    assert all(record["checks"].values()) and not observer.incomplete and not observer.support_incomplete
    row = next(row for row in record["rows"] if row["identity"][1:] == [axis, 1])
    summary, witness = row["summary"], row["summary"]["witness"]
    a, b = [column for column in range(3) if column != axis]
    tangent = components[a] + components[b]
    balance = 2 * math.sqrt(components[a] * components[b]) / max(tangent, 1e-6)
    fraction = tangent / max(sum(components), 1e-6)
    reliability = tangent / (tangent + 24)
    assert witness["components"] == list(components)
    assert witness["normal_score"] == float(graph.plane_scores[0, axis])
    assert tuple(witness[key] for key in ("balance", "normal_fraction", "reliability", "derived_score")) == pytest.approx(
        (balance, fraction, reliability, balance * fraction * reliability))
    assert summary["component_mass"] == [5 * value for value in components]
    assert summary["component_nonzero"] == [5 * int(value > 0) for value in components]
    assert summary["tangent_nonzero"] == [5 * int(components[column] > 0) for column in (a, b)]
    assert summary["tangent_cooccurrence"] == 5 * int(components[a] > 0 and components[b] > 0)
    assert summary["support_patterns"] == [5 * int(code == pattern) for code in range(8)]
    seed = next(row for row in attrition["stages"]["seed"]["terminal"] if row["identity"][1:] == [axis, 1])
    assert seed["seed_predicates"]["after_score_count"] == 5 * int(score_pass)
    assert (seed["eligible_seed_count"] > 0) is score_pass
    assert outputs[(axis, 1)] == subject.region_growth._seed_and_grow(graph, axis, 1, subject.region_growth.DEFAULT_REGION_GROWTH_CONFIG)
    assert next(row for row in record["rows"] if row["identity"][1:] == [axis, -1])["summary"]["witness"] is None


def test_support_score_patterns_distinguish_global_mass_from_tangent_colocation():
    graph = _support_graph(((0, 80, 0), (0, 0, 80), (0, 0, 0), (0, 0, 0), (0, 0, 0)))
    _observer, _capture, _attrition, record, _outputs = _observe_support(graph)
    summary = next(row["summary"] for row in record["rows"] if row["identity"][1:] == [0, 1])
    assert summary["component_mass"] == [0, 80, 80]
    assert summary["tangent_nonzero"] == [1, 1] and summary["tangent_cooccurrence"] == 0
    assert summary["support_patterns"] == [3, 0, 1, 0, 1, 0, 0, 0]
    assert summary["witness"]["normal_score"] == summary["witness"]["balance"] == 0


@pytest.mark.parametrize("permutation", [(0, 1, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)])
def test_support_score_actual_axis_permutation_and_side_reflection(permutation):
    import numpy as np
    graph = _support_graph(((0, 40, 40), (40, 0, 40), (40, 40, 0), (240, 40, 40), (0, 1, 1)),
                           sides=np.array([[1, -1, 1], [-1, 1, 1], [1, 1, -1], [-1, -1, 1], [0, 0, 0]], dtype=np.int8))
    changed = _support_graph(graph.supports[:, permutation], sides=-graph.sides[:, permutation])
    original = subject._support_score_rows("image-0000", graph)
    transformed = subject._support_score_rows("image-0000", changed)
    for new_axis, old_axis in enumerate(permutation):
        for side in (-1, 1):
            before = next(row["summary"] for row in original if row["identity"][1:] == [old_axis, side])
            after = next(row["summary"] for row in transformed if row["identity"][1:] == [new_axis, -side])
            assert after["side_count"] == before["side_count"]
            for field in ("component_mass", "component_nonzero"):
                assert after[field] == [before[field][column] for column in permutation]
            expected_patterns = [0] * 8
            for code, count in enumerate(before["support_patterns"]):
                mapped = sum(1 << column for column, previous in enumerate(permutation) if code & (1 << previous))
                expected_patterns[mapped] = count
            assert after["support_patterns"] == expected_patterns
            assert after["tangent_cooccurrence"] == before["tangent_cooccurrence"]
            if before["witness"] is not None:
                assert after["witness"]["components"] == [before["witness"]["components"][column] for column in permutation]
                for field in ("normal_score", "balance", "normal_fraction", "reliability", "derived_score"):
                    assert after["witness"][field] == pytest.approx(before["witness"][field])


@pytest.mark.parametrize("boundary", [True, False], ids=["source-equality", "source-next-float-below"])
def test_support_score_actual_score_threshold_boundary_without_support_promotion(boundary):
    import numpy as np
    threshold = np.float32(subject.region_growth.DEFAULT_REGION_GROWTH_CONFIG.min_seed_score)
    score = threshold if boundary else np.nextafter(threshold, np.float32(-np.inf))
    total = 24 * float(score) / (1 - float(score))
    graph = _support_graph((0, total / 2, total / 2))
    assert graph.plane_scores[0, 0] == score
    _observer, _capture, attrition, record, _outputs = _observe_support(graph)
    seed = next(row for row in attrition["stages"]["seed"]["terminal"] if row["identity"][1:] == [0, 1])
    assert seed["seed_predicates"]["after_score_count"] == 5 * int(boundary)
    assert seed["eligible_seed_count"] == 0  # Reliability boundary needs <12 support; no relaxed gate.
    assert record["status"] == "complete"


def test_support_score_same_node_witness_and_scalar_redaction():
    graph = _support_graph(((0, 40, 40), (240, 40, 40), (0, 1, 79), (0, 1, 1), (0, 0, 0)))
    _observer, _capture, _attrition, record, _outputs = _observe_support(graph)
    summary = next(row["summary"] for row in record["rows"] if row["identity"][1:] == [0, 1])
    assert summary["witness"]["components"] == [0, 40, 40]
    assert summary["component_mass"] == [240, 82, 160]  # Maxima/mass are not substituted into the witness.
    payload = json.loads(subject.canonical_bytes(record))
    assert payload == record
    def check_scalars(value):
        if type(value) is dict:
            for item in value.values(): check_scalars(item)
        elif type(value) is list:
            for item in value: check_scalars(item)
        else: assert type(value) in (str, int, float, bool, type(None))
    check_scalars(record)
    assert all(word not in subject.canonical_bytes(record) for word in (b'"pixels"', b'"coordinates"', b'"labels"', b'"node_xy"', b'.png', b'private'))
    assert all(type(value) in (str, int, bool, type(None), list) for value in _observer.support_frames["image-0000"].values())


@pytest.mark.parametrize("change", ["bool-count", "negative-mass", "nonfinite", "patterns", "positive-count", "tangent-count", "cooccurrence",
                                       "wrong-factor", "wrong-source-score", "wrong-triplet", "extra", "missing", "foreign-id", "array", "huge-nodes"])
def test_support_score_malformed_rows_refuse(change):
    import numpy as np
    row = deepcopy(subject._support_score_rows("image-0000", _support_graph((0, 40, 40)))[1])
    summary = row["summary"]
    if change == "bool-count": summary["side_count"] = True
    elif change == "negative-mass": summary["component_mass"][0] = -1
    elif change == "nonfinite": summary["witness"]["reliability"] = float('inf')
    elif change == "patterns": summary["support_patterns"][0] += 1
    elif change == "positive-count": summary["component_nonzero"][0] = 1
    elif change == "tangent-count": summary["tangent_nonzero"][0] = 4
    elif change == "cooccurrence": summary["tangent_cooccurrence"] = 4
    elif change == "wrong-factor": summary["witness"]["balance"] = 0.2
    elif change == "wrong-source-score": summary["witness"]["normal_score"] = 0.2
    elif change == "wrong-triplet": summary["witness"]["components"] = [0, 200, 200]
    elif change == "extra": summary["pixels"] = []
    elif change == "missing": del summary["support_patterns"]
    elif change == "foreign-id": row["identity"][0] = '../private/input.png'
    elif change == "array": summary["component_mass"] = np.array(summary["component_mass"])
    elif change == "huge-nodes": summary["node_count"] = subject.SUPPORT_NODES + 1
    with pytest.raises(ValueError): subject._support_score_row(row)


@pytest.mark.parametrize("change", ["shape", "dtype", "nan", "negative", "wrong-source-score", "huge"])
def test_support_score_invalid_graph_is_diagnostic_only(change):
    import numpy as np
    graph = _support_graph((0, 40, 40))
    if change == "shape": graph = dataclasses.replace(graph, supports=graph.supports[:, :2])
    elif change == "dtype": graph = dataclasses.replace(graph, supports=graph.supports.astype(bool))
    elif change == "nan": graph = dataclasses.replace(graph, supports=np.full_like(graph.supports, np.nan))
    elif change == "negative": graph = dataclasses.replace(graph, supports=-graph.supports)
    elif change == "wrong-source-score": graph = dataclasses.replace(graph, plane_scores=np.zeros_like(graph.plane_scores))
    elif change == "huge": graph = SimpleNamespace(supports=np.zeros((subject.SUPPORT_NODES + 1, 3)),
        plane_scores=np.zeros((subject.SUPPORT_NODES + 1, 3)), sides=np.ones((subject.SUPPORT_NODES + 1, 3), dtype=np.int8))
    observer = subject.ReadingAttritionObserver(); observer.support_begin("image-0000")
    with subject.region_growth._scalar_attrition_context(observer, frame="image-0000"):
        observer.support_graph("image-0000", (graph, {}))
    assert observer.support_incomplete and observer.support_frames["image-0000"]["rows"] is None


@pytest.mark.parametrize("change", ["missing", "duplicate", "foreign", "seed-count", "seed-maximum", "item-cap", "byte-cap"])
def test_support_score_readback_joins_and_combined_envelope_caps(change):
    _observer, _capture, attrition, record, _outputs = _observe_support(_support_graph((0, 40, 40)))
    attrition, record = deepcopy(attrition), deepcopy(record)
    if change == "missing": record["rows"].pop()
    elif change == "duplicate": record["rows"].append(deepcopy(record["rows"][0]))
    elif change == "foreign": record["rows"][0]["identity"][0] = 'image-0001'
    elif change == "seed-count": attrition["stages"]["seed"]["terminal"][0]["seed_predicates"]["node_count"] += 1
    elif change == "seed-maximum": attrition["stages"]["seed"]["terminal"][1]["seed_predicates"]["max_side_normal_score"] = 0.5
    elif change == "item-cap": record["limits"]["items"] = attrition["limits"]["items"] = 6
    elif change == "byte-cap": record["limits"]["bytes"] = attrition["limits"]["bytes"] = len(subject.canonical_bytes(record))
    if change in ("duplicate", "foreign", "item-cap", "byte-cap"):
        with pytest.raises(ValueError): subject._support_score_account(record, attrition)
    else:
        _checks, status = subject._support_score_account(record, attrition)
        assert status == 'incomplete'


@pytest.mark.parametrize("limit", ["items", "bytes"])
def test_support_score_observer_cap_preserves_original_seed_outputs(limit):
    graph = _support_graph((0, 40, 40))
    original = subject.region_growth._seed_and_grow(graph, 0, 1, subject.region_growth.DEFAULT_REGION_GROWTH_CONFIG)
    observer = subject.ReadingAttritionObserver(**{'max_' + limit: 1})
    observer, _capture, _attrition, record, outputs = _observe_support(graph, observer=observer)
    assert outputs[(0, 1)] == original and observer.support_incomplete
    assert record is None or record['status'] == 'incomplete'


@pytest.mark.parametrize("mode", ['normal', 'observer-error', 'build-error', 'measure-error', 'repeat', 'foreign-context'])
def test_support_score_wrapper_one_original_call_forwarding_identity_and_restoration(monkeypatch, mode):
    graph = _support_graph((0, 40, 40)); build_result = (graph, {'discarded': object()})
    frame = SimpleNamespace(frame_id='image-0000'); original_result = object(); argument = object(); keyword = object()
    original_error = RuntimeError('private path must not enter telemetry')
    calls = []; observer = subject.ReadingAttritionObserver()
    def original_build(*args, **kwargs):
        calls.append((args, kwargs))
        if mode == 'build-error': raise original_error
        return build_result
    def original_measure(received, *, token):
        assert received is frame and token is keyword
        with subject.region_growth._scalar_attrition_context(observer, frame='image-0001' if mode == 'foreign-context' else frame.frame_id):
            assert subject.region_growth._build_graph(argument, keyword=keyword) is build_result
            if mode == 'repeat': assert subject.region_growth._build_graph(argument, keyword=keyword) is build_result
        if mode == 'measure-error': raise original_error
        return original_result
    monkeypatch.setattr(subject.region_growth, '_build_graph', original_build)
    monkeypatch.setattr(subject.lattice_measure, '_measure_frame', original_measure)
    if mode == 'observer-error':
        monkeypatch.setattr(observer, 'support_graph', lambda *args: (_ for _ in ()).throw(ValueError('observer failed')))
    capture = subject.GraphCapture(observer)
    if mode in ('build-error', 'measure-error'):
        with pytest.raises(RuntimeError) as error: capture.invoke(lambda: subject.lattice_measure._measure_frame(frame, token=keyword))
        assert error.value is original_error
    else:
        assert capture.invoke(lambda: subject.lattice_measure._measure_frame(frame, token=keyword)) is original_result
    assert len(calls) == (2 if mode == 'repeat' else 1)
    assert all(args == (argument,) and kwargs == {'keyword': keyword} for args, kwargs in calls)
    assert subject.region_growth._build_graph is original_build and subject.lattice_measure._measure_frame is original_measure
    assert subject.region_growth._attrition_state.get() is None and subject.region_growth._attrition_frame.get() == ''
    assert observer.support_incomplete is (mode != 'normal')
    rows = observer.support_frames[frame.frame_id]['rows']
    assert rows is None or all(type(row) is dict for row in rows)


def test_support_score_not_reached_is_explicit_null_and_missing_is_incomplete(monkeypatch):
    observer = subject.ReadingAttritionObserver(); frame = SimpleNamespace(frame_id='image-0000'); sentinel = object()
    def early_return(received):
        assert received is frame
        for axis in range(3):
            for side in (-1, 1):
                identity = (frame.frame_id, axis, side)
                observer.expect('seed', identity)
                observer.finish('seed', identity, 'not_reached', axis=axis, side=side, eligible_seed_count=None)
        return sentinel
    monkeypatch.setattr(subject.lattice_measure, '_measure_frame', early_return)
    monkeypatch.setattr(subject.region_growth, '_build_graph', lambda *args: pytest.fail('unreached build'))
    capture = subject.GraphCapture(observer)
    assert capture.invoke(lambda: subject.lattice_measure._measure_frame(frame)) is sentinel
    capture.graph = SimpleNamespace(frame_ids=(frame.frame_id,));capture.arguments=((),dict(lattice_result=SimpleNamespace(readings=())))
    attrition = subject._reading_attrition_record(observer, capture)
    record = subject._support_score_record(observer, capture, attrition)
    assert record['status'] == 'complete' and all(row['state'] == 'not_reached' and row['summary'] is None for row in record['rows'])
    observer.support_frames.clear()
    missing = subject._support_score_record(observer, capture, attrition)
    assert missing['status'] == 'incomplete' and all(row['state'] == 'missing' and row['summary'] is None for row in missing['rows'])


@pytest.mark.parametrize("built", [True, False], ids=["measured-before-seed-loop", "graph-unavailable"])
def test_support_score_graph_return_before_seed_loop_preserves_stage_distinction(built):
    observer = subject.ReadingAttritionObserver(); frame = 'image-0000'
    observer.support_begin(frame)
    with subject.region_growth._scalar_attrition_context(observer, frame=frame):
        observer.support_graph(frame, (_support_graph((0, 0, 0)) if built else None, {}))
    for axis in range(3):
        for side in (-1, 1):
            identity = (frame, axis, side)
            observer.expect('seed', identity)
            observer.finish('seed', identity, 'not_reached', axis=axis, side=side, eligible_seed_count=None)
    capture = subject.GraphCapture(observer)
    capture.graph = SimpleNamespace(frame_ids=(frame,))
    capture.arguments = ((), dict(lattice_result=SimpleNamespace(readings=())))
    attrition = subject._reading_attrition_record(observer, capture)
    record = subject._support_score_record(observer, capture, attrition)
    assert attrition['status'] == record['status'] == 'complete'
    assert all(row['disposition'] == 'not_reached' and row['seed_predicates'] is None
               for row in attrition['stages']['seed']['terminal'])
    assert all(row['state'] == ('measured' if built else 'graph_unavailable') for row in record['rows'])
    assert all((row['summary'] is not None) is built for row in record['rows'])


def test_support_score_report_digest_and_graph_payload_unchanged(monkeypatch, tmp_path):
    graph, sources = controlled_graph('unsupported_reading_endpoint')
    off = _run(tmp_path, monkeypatch, graph, sources)
    observer = subject.ReadingAttritionObserver(); atom_graph = _support_graph((0, 40, 40))
    for frame in graph.frame_ids:
        _observe_support(atom_graph, frame=frame, observer=observer)
    # Authored reader/graph seam; the actual score and seed producers above are
    # not replaced, and no claim is made that this graph came from image inference.
    _emit_attrition(observer, sources['lattice_result'].readings, emit_seeds=False)
    capture = capture_for(graph, sources);capture.reading_attrition=observer
    on = subject.build_report(capture, deepcopy(off.report['run']))
    assert on['status'] == 'complete' and on['support_score_attrition']['status'] == 'complete'
    assert on['graph'] == off.report['graph'] and graph.result_sha256 == off.production_result.result_sha256
    assert on['result_sha256'] != off.report['result_sha256']
    changed = deepcopy(on); changed['support_score_attrition']['rows'][1]['summary']['witness']['balance'] = 0.1
    with pytest.raises(subject.DiagnosticError, match='digest'):subject.validate_report(changed)
    changed.pop("result_sha256")
    subject._seal(changed)
    with pytest.raises(subject.DiagnosticError, match='support_score_attrition'):subject.validate_report(changed)
    observer.support_incomplete=True
    incomplete=subject.build_report(capture,deepcopy(off.report['run']))
    assert incomplete['status']=='incomplete' and incomplete['reason']=='support_score_attrition_incomplete'
    assert incomplete['graph']==off.report['graph']


def _paired_fixture(monkeypatch, *, case="narrow", frame="image-0000", observer=None,
                    failure=None, mutation=None, emit_seeds=True, weights=None,
                    yaw=180, lines=None, residuals=None, support_line_weight_policy="structural"):
    import numpy as np
    from test_region_growth import _raster_controls, _calibration
    rg=subject.region_growth
    labels=np.zeros((97,97),np.int32);left=23 if case=="phase" else 19
    if case != "single": labels[16:81,left:left+3]=1
    grey=np.zeros(labels.shape,np.uint8);usable=np.ones(labels.shape,bool)
    if case=="mask-hole": usable[32,left+1]=False
    y=15 if case=="boundary" else 32
    if lines is None: lines=[] if case=="empty" else [(8,y,88,y),(left+1,8,left+1,88)]
    if residuals is None: residuals=[[100,100,0],[100,0,100]]
    calls=_raster_controls(monkeypatch,lines,residuals=None if not lines else residuals,
                           weights=weights)
    cal=_calibration(yaw=yaw,width=97,height=97);config=rg.DEFAULT_REGION_GROWTH_CONFIG
    lab=np.zeros((*labels.shape,3),np.float32)
    owner=subject.ReadingAttritionObserver(paired_segment_support=True) if observer is None else observer
    original_build=rg._build_graph; original_support=rg._segment_support
    sentinel=object();original_error=RuntimeError("source failure")
    graph_result=[]
    def measure(value):
        assert value.frame_id==frame
        with rg._scalar_attrition_context(owner,frame=frame):
            if failure=="before-build": raise original_error
            result=rg._build_graph(labels,grey,lab,usable,cal,config)
            graph_result.append(result)
            if mutation is not None: mutation(owner.paired,result[0])
            if failure=="after-build": raise original_error
            if emit_seeds:
                for axis in range(3):
                    for side in (-1,1): rg._seed_and_grow(result[0],axis,side,config)
        return sentinel
    monkeypatch.setattr(subject.lattice_measure,"_measure_frame",measure)
    capture=subject.GraphCapture(owner,support_line_weight_policy=support_line_weight_policy)
    if failure:
        with pytest.raises(RuntimeError) as caught:
            capture.invoke(lambda:subject.lattice_measure._measure_frame(SimpleNamespace(frame_id=frame)))
        assert caught.value is original_error
    else:
        assert capture.invoke(lambda:subject.lattice_measure._measure_frame(SimpleNamespace(frame_id=frame))) is sentinel
    assert rg._build_graph is original_build and rg._segment_support is original_support
    assert subject.lattice_measure._measure_frame is measure
    assert rg._attrition_state.get() is None and rg._attrition_frame.get()==""
    assert owner.paired.state is None and owner.paired.frame is None
    capture.graph=SimpleNamespace(frame_ids=(frame,))
    capture.arguments=((),dict(lattice_result=SimpleNamespace(readings=())))
    return owner,capture,calls,graph_result


def _paired_record(owner,capture):
    attrition=subject._reading_attrition_record(owner,capture)
    return subject._paired_support_record(owner,capture,attrition),attrition


@pytest.mark.parametrize("case,decision",[("narrow","continue_candidate"),("phase","stop_no_new_only"),
                                         ("boundary","stop_no_new_only"),("empty","stop_no_new_only")])
def test_paired_actual_one_source_pass_target_partition_and_mass(monkeypatch,case,decision):
    owner,capture,calls,results=_paired_fixture(monkeypatch,case=case)
    record,attrition=_paired_record(owner,capture)
    assert record["status"]=="complete" and all(record["checks"].values())
    assert record["decision"]==decision and len(record["rows"])==6
    assert calls==dict(enhance=1,detect=1,clip=int(case!="empty"),residual=int(case!="empty"),weight=int(case!="empty"))
    source=results[0][0].supports
    assert source.dtype==(subject.region_growth.np.float64 if case=="empty" else subject.region_growth.np.float32)
    target=next(row for row in record["rows"] if row["axis"]==0 and row["side"]==-1)
    if case=="narrow":
        assert target["partition"]["new_only"]==1 and target["globally_new"]==1
        assert target["witness"]["old"]["support"][2]==0
        assert all(target["witness"]["new"]["passes"])
        assert record["qualifying_targets"]==[["image-0000",0,-1]]
        assert record["frame_rows"][0]["union"]["new_only"]==1
    assert record["causal_scope"]=="combined_centerline_and_usable_mask_effect_only"
    assert subject._pair_account(record,attrition)==(record["checks"],record["status"])
    encoded=subject.canonical_bytes(record)
    assert b'node_id' not in encoded and b'labels' not in encoded and b'coordinates' not in encoded


def test_paired_non_target_gains_do_not_qualify(monkeypatch):
    owner,capture,_,_=_paired_fixture(monkeypatch,frame="image-0001")
    record,_=_paired_record(owner,capture)
    assert record["decision"]=="stop_non_target_only" and not record["qualifying_targets"]
    assert any(row["partition"]["new_only"] for row in record["rows"])


def test_paired_mask_holes_keep_old_credit_and_narrow_causal_scope(monkeypatch):
    owner,capture,_,_=_paired_fixture(monkeypatch,case="mask-hole")
    record,_=_paired_record(owner,capture)
    mass=record["frame_rows"][0]["mass"]
    assert mass["old_unsupported"]==[0.,0.,0.]
    assert mass["new_unsupported"][2]>0 and mass["new_unsupported"][1]>0
    assert record["causal_scope"]=="combined_centerline_and_usable_mask_effect_only"


def test_paired_post_clip_evidence_census_separates_raw_weighted_and_mask_loss(monkeypatch):
    owner,capture,_,_=_paired_fixture(monkeypatch,case="mask-hole",weights=[.25,.5])
    record,attrition=_paired_record(owner,capture)
    frame=record["frame_rows"][0]
    evidence=frame["census"]["evidence"]
    assert evidence["stage"]=="post_clipping_retained_evidence_pre_allocation"
    assert evidence["dense_invalid_scope"]=="dense_centerline_invalid_label_or_usable_mask_samples"
    assert evidence["nonzero_associated_segments"]==[0,1,1]
    assert evidence["length_association_mass"]==[0.,80.,80.]
    assert evidence["length_weight_association_mass"]==[0.,40.,20.]
    assert evidence["dense_invalid_length_association_mass"]==[0.,80./81.,80./81.]
    assert evidence["dense_invalid_length_weight_association_mass"]==[0.,40./81.,20./81.]
    assert frame["mass"]["potential"]==evidence["length_weight_association_mass"]
    assert frame["mass"]["new_unsupported"]==evidence["dense_invalid_length_weight_association_mass"]
    assert subject._pair_account(record,attrition)==(record["checks"],"complete")


@pytest.mark.parametrize("case",["weight-over-raw","invalid-over-raw","valid-over-raw"])
def test_paired_evidence_validator_rejects_impossible_weighted_mass(case):
    evidence=dict(schema=subject.PAIR_EVIDENCE_SCHEMA,stage=subject.PAIR_EVIDENCE_STAGE,
        dense_invalid_scope=subject.PAIR_INVALID_SCOPE,nonzero_associated_segments=[1,1,1],
        length_association_mass=[2.,3.,4.],length_weight_association_mass=[1.,2.,3.],
        dense_invalid_length_association_mass=[.5,1.,1.5],
        dense_invalid_length_weight_association_mass=[.25,.5,.75])
    census=dict(retained=3,zero=0)
    mass=dict(potential=evidence["length_weight_association_mass"],
              new_unsupported=evidence["dense_invalid_length_weight_association_mass"],
              allowance=[1e-8,1e-8,1e-8])
    subject._pair_validate_evidence(evidence,census,mass)
    broken=deepcopy(evidence)
    if case=="weight-over-raw": broken["length_association_mass"][0]=.5
    elif case=="invalid-over-raw": broken["dense_invalid_length_association_mass"][0]=.1
    else:
        broken["length_association_mass"][0]=1.4
        broken["dense_invalid_length_association_mass"][0]=.5
        broken["length_weight_association_mass"][0]=1.35
        broken["dense_invalid_length_weight_association_mass"][0]=.4
        mass["potential"][0]=1.35
        mass["new_unsupported"][0]=.4
    with pytest.raises(ValueError,match="paired evidence mass bounds"):
        subject._pair_validate_evidence(broken,census,mass)


@pytest.mark.parametrize("failure",["before-build","after-build"])
def test_paired_wrapper_source_exception_and_restoration(monkeypatch,failure):
    owner,capture,_,_=_paired_fixture(monkeypatch,failure=failure)
    assert owner.paired.state is None
    if failure=="before-build": assert owner.paired.frames["image-0000"]["state"]=="not_reached"
    else: assert owner.paired.frames["image-0000"]["state"]=="measured"


def _paired_direct_begin(monkeypatch, *, labels=None, owner=None):
    import numpy as np
    from test_region_growth import _calibration
    rg=subject.region_growth
    owner=subject.ReadingAttritionObserver(paired_segment_support=True) if owner is None else owner
    owner.paired.enter_frame("image-0000")
    labels=np.zeros((2,81),np.int32) if labels is None else labels
    grey=np.zeros(labels.shape,np.uint8);usable=np.ones(labels.shape,bool)
    segments=np.asarray([[0.,0.,80.,0.],[0.,1.,80.,1.]])
    lengths=np.asarray([80.,80.]);weights=np.asarray([1.,.5]);association=np.asarray([[1.,0.,0.],[0.,0.,0.]])
    values=(grey,usable,labels,_calibration(),rg.DEFAULT_REGION_GROWTH_CONFIG,segments,lengths,weights,association,int(labels.max())+1,(1.,1.))
    with rg._scalar_attrition_context(owner,frame="image-0000"): owner.paired.begin(*values)
    return owner,values


@pytest.mark.parametrize("fault",["duplicate-begin","missing","order","foreign","duplicate","modified-label", "modified-weight","bad-final","double-final","wrong-return","changed-return","double-return","wrong-context"])
def test_paired_lifecycle_invalid_joins_clear_all_transients(monkeypatch,fault):
    import numpy as np
    import weakref
    rg=subject.region_growth;owner,values=_paired_direct_begin(monkeypatch)
    pair=owner.paired;s=pair.state
    refs=[weakref.ref(s["raster"]),weakref.ref(s["workspace"])]
    ids=np.zeros(11,np.int32);segments,lengths,weights,association=values[5:9]
    support=np.asarray([[80.,0.,0.]],np.float32)
    with rg._scalar_attrition_context(owner,frame="foreign" if fault=="wrong-context" else "image-0000"):
        with pytest.raises(ValueError):
            if fault=="duplicate-begin": pair.begin(*values)
            elif fault=="missing": pair.finalize(support)
            elif fault=="order": pair.segment(1,segments,lengths,weights,association,None,None)
            elif fault=="foreign": pair.segment(0,segments.copy(),lengths,weights,association,ids,11)
            else:
                pair.segment(0,segments,lengths,weights,association,ids,11)
                if fault=="duplicate": pair.segment(0,segments,lengths,weights,association,ids,11)
                pair.segment(1,segments,lengths,weights,association,None,None)
                if fault=="modified-label": values[2][0,0]=-1
                if fault=="modified-weight": weights[1]=.7
                if fault=="bad-final": support=np.zeros((2,3),np.float32)
                pair.finalize(support)
                if fault=="double-final": pair.finalize(support)
                if fault=="wrong-return": pair.returned((support.copy(),{}))
                if fault=="changed-return": support[0,0]=79
                pair.returned((support,{}))
                if fault=="double-return": pair.returned((support,{}))
    assert pair.incomplete and pair.state is None and not s
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("method",["begin","segment","finalize","returned","graph"])
@pytest.mark.parametrize("exception",[RuntimeError("observer failed"),KeyboardInterrupt(),SystemExit(9)])
def test_paired_callback_failures_cleanup_and_preserve_source_when_applicable(monkeypatch,method,exception):
    owner=subject.ReadingAttritionObserver(paired_segment_support=True)
    original=getattr(owner.paired,method)
    def fail(*args,**kwargs):
        if method=="begin": original(*args,**kwargs)
        raise exception
    monkeypatch.setattr(owner.paired,method,fail)
    if isinstance(exception,Exception):
        owner,capture,calls,_=_paired_fixture(monkeypatch,observer=owner)
        assert owner.paired.incomplete
        assert calls["detect"]==1
    else:
        with pytest.raises(type(exception)) as caught: _paired_fixture(monkeypatch,observer=owner)
        assert caught.value is exception
    assert owner.paired.state is None


@pytest.mark.parametrize("limit",["items","bytes","time"])
def test_paired_caps_fail_closed_preserve_source_and_cleanup(monkeypatch,limit):
    owner=subject.ReadingAttritionObserver(paired_segment_support=True,
        **({"max_"+limit:1} if limit!="time" else {}))
    if limit=="time": owner.paired.total_seconds=181
    owner,capture,calls,_=_paired_fixture(monkeypatch,observer=owner)
    assert owner.paired.incomplete and owner.paired.state is None and calls["detect"]==1


def test_paired_graph_not_reached_is_unmeasured(monkeypatch):
    owner=subject.ReadingAttritionObserver(paired_segment_support=True)
    frame=SimpleNamespace(frame_id="image-0000")
    monkeypatch.setattr(subject.lattice_measure,"_measure_frame",lambda received:object())
    capture=subject.GraphCapture(owner)
    capture.invoke(lambda:subject.lattice_measure._measure_frame(frame))
    capture.graph=SimpleNamespace(frame_ids=(frame.frame_id,));capture.arguments=((),dict(lattice_result=SimpleNamespace(readings=())))
    record,attrition=_paired_record(owner,capture)
    assert record["status"]=="incomplete" and record["decision"]=="inconclusive"
    assert all(r["state"]=="not_reached" and r["partition"] is None for r in record["rows"])


def test_paired_distinct_union_deduplicates_overlapping_marginals_and_transfers():
    import numpy as np
    rg=subject.region_growth
    old=np.asarray([[0,48,48],[48,0,48],[0,0,0]],np.float32)
    new=np.asarray([[48,0,48],[48,0,48],[0,48,48]],np.float32)
    # Source-score domains bound; setting zero margin deliberately allows
    # overlapping authored hypotheses only for this synthetic union control.
    graph=SimpleNamespace(supports=old,plane_scores=rg._normal_scores(old),sides=np.ones((3,3),np.int8))
    rows,union,_=subject._pair_project("image-0001",graph,new,rg.DEFAULT_REGION_GROWTH_CONFIG)
    assert union==dict(both=2,old_only=0,new_only=1,neither=0)
    assert any(r["transferred"]==1 for r in rows)
    first=np.asarray([True,True,False]);second=np.asarray([True,False,True])
    aggregate_old=np.logical_or(first,first);aggregate_new=np.logical_or(second,second)
    assert np.count_nonzero(aggregate_old)==2 != 2*np.count_nonzero(first)
    assert subject._pair_partition(aggregate_old,aggregate_new)==dict(both=1,old_only=1,new_only=1,neither=0)


@pytest.mark.parametrize("field",["partition","coverage","union","losses","witness","decision","mass","ordinal","transfer","source-seed"])
def test_paired_scalar_tampering_even_resealed_refuses(monkeypatch,field):
    owner,capture,_,_=_paired_fixture(monkeypatch)
    record,attrition=_paired_record(owner,capture);record=deepcopy(record);attrition=deepcopy(attrition)
    target=next(r for r in record["rows"] if r["witness"] is not None)
    if field=="partition": target["partition"]["new_only"]+=1
    if field=="coverage": target["coverage"]="zero_to_zero"
    if field=="union": record["frame_rows"][0]["union"]["neither"]+=1
    if field=="losses": record["losses"]=[{}]
    if field=="witness": target["witness"]["old"]["support"]=target["witness"]["new"]["support"]
    if field=="decision": record["decision"]="stop_no_new_only"
    if field=="mass": record["frame_rows"][0]["mass"]["new"][0]+=1
    if field=="ordinal": record["frame_rows"][0]["census"]["observed"]-=1
    if field=="transfer": target["transferred"]+=1
    if field=="source-seed": attrition["stages"]["seed"]["terminal"][-1]["eligible_seed_count"]+=1
    with pytest.raises((ValueError,TypeError,KeyError)): subject._pair_account(record,attrition)


def test_paired_report_optional_digest_and_incomplete_envelope(monkeypatch):
    owner,capture,_,_=_paired_fixture(monkeypatch)
    pair,attrition=_paired_record(owner,capture)
    base=dict(schema=subject.SCHEMA,producer=subject.PRODUCER,status="incomplete",reason="upstream",field="",
              g_calls=0,graph=None,run={},notes=[],reading_attrition=attrition,paired_segment_support=pair)
    report=subject._seal(base);subject.validate_report(report)
    changed=deepcopy(report);changed["paired_segment_support"]["decision"]="bad"
    with pytest.raises(subject.DiagnosticError,match="digest"):subject.validate_report(changed)
    changed.pop("result_sha256");subject._seal(changed)
    with pytest.raises(subject.DiagnosticError,match="paired_segment_support"):subject.validate_report(changed)
    owner.paired.discard("observer_error")
    pair,attrition=_paired_record(owner,capture)
    base.update(paired_segment_support=pair,status="incomplete",reason="paired_segment_support_incomplete",field="observer")
    base.pop("result_sha256")
    subject.validate_report(subject._seal(base))


@pytest.mark.parametrize("field", ["score", "margin", "support"])
@pytest.mark.parametrize("offset", [-1, 0, "quarter-ulp", 1])
def test_paired_witness_thresholds_match_native_source_boundary_and_nextafter(field,offset):
    import numpy as np
    from test_region_growth import _seed_predicate_graph
    rg=subject.region_growth
    old=np.zeros((1,3),np.float32);new=np.asarray([[0.,50.,50.]],np.float32)
    os,ns=rg._normal_scores(old),rg._normal_scores(new)
    values=[ns[0,0],ns[0,0]-ns[0,1],new[0,1:].sum()]
    index={"score":0,"margin":1,"support":2}[field];base=values[index]
    above=np.nextafter(base,np.float32(np.inf));below=np.nextafter(base,np.float32(-np.inf))
    threshold=(float(below) if offset==-1 else float(above) if offset==1 else
               float(base)+(float(above)-float(base))/4 if offset=="quarter-ulp" else float(base))
    changes=dict(zip(("min_seed_score","min_seed_margin","min_seed_support"),(0.,0.,0.)))
    changes[("min_seed_score","min_seed_margin","min_seed_support")[index]]=threshold
    config=dataclasses.replace(rg.DEFAULT_REGION_GROWTH_CONFIG,**changes)
    sides=np.ones((1,3),np.int8)
    joint=subject._pair_metrics(new,ns,sides,0,1,config)[0]
    assert joint.tolist()==[offset!=1]
    # Original seed hook must agree, including float32 array threshold casting.
    graph=dataclasses.replace(_seed_predicate_graph(),node_area=np.ones(1,np.int32),
        node_lab=np.zeros((1,3),np.float32),node_texture=np.zeros(1,np.float32),
        supports=new,plane_scores=ns,sides=sides,axis_line_dirs=np.zeros((1,3,2),np.float32),adjacency=((),))
    observer=subject.ReadingAttritionObserver()
    with rg._scalar_attrition_context(observer,frame="image-0000"):
        rg._seed_and_grow(graph,0,1,config)
    assert list(observer.scalar_rows["seed"].values())[0]["eligible_seed_count"]==int(joint[0])
    if offset==1:
        with pytest.raises(ValueError,match="membership"):
            subject._pair_witness(0,old,new,os,ns,0,1,sides,config)
    else:
        witness=subject._pair_witness(0,old,new,os,ns,0,1,sides,config)
        subject._pair_validate_witness(witness,list(changes.values()),0,1)
        assert all(witness["new"]["passes"])
        if offset=="quarter-ulp":
            assert float(base)<threshold  # Python-float comparison would falsely fail.
        bad=deepcopy(witness);bad["new"]["passes"][index]=False
        with pytest.raises(ValueError): subject._pair_validate_witness(bad,list(changes.values()),0,1)


def test_paired_measured_support_without_seed_terminal_is_inconclusive(monkeypatch):
    owner,capture,_,_=_paired_fixture(monkeypatch,emit_seeds=False)
    record,attrition=_paired_record(owner,capture)
    assert all(row["disposition"]=="not_reached" for row in attrition["stages"]["seed"]["terminal"])
    assert all(row["state"]=="measured" for row in record["rows"])
    assert any(row["witness"] is not None for row in record["rows"])
    assert not record["checks"]["source_seed_join"]
    assert record["status"]=="incomplete" and record["decision"]=="inconclusive"
    assert record["qualifying_targets"]==[]
    subject._pair_account(record,attrition)


@pytest.mark.parametrize("field",["axis","side","target-axis","target-side","row-bytes"])
def test_paired_resealed_identity_and_row_caps_refused(monkeypatch,field):
    owner,capture,_,_=_paired_fixture(monkeypatch)
    pair,attrition=_paired_record(owner,capture)
    if field=="axis": pair["rows"][0]["axis"]=False
    if field=="side": next(r for r in pair["rows"] if r["side"]==1)["side"]=True
    if field=="target-axis": pair["qualifying_targets"][0][1]=False
    if field=="target-side": pair["qualifying_targets"]=[["image-0000",2,True]]
    if field=="row-bytes": pair["rows"][0]["state"]="x"*2049
    report=subject._seal(dict(schema=subject.SCHEMA,producer=subject.PRODUCER,status="incomplete",
        reason="upstream",field="",g_calls=0,graph=None,run={},notes=[],
        reading_attrition=attrition,paired_segment_support=pair))
    with pytest.raises(subject.DiagnosticError,match="paired_segment_support"):
        subject.validate_report(report)


def test_paired_producer_row_byte_cap_is_enforced(monkeypatch):
    monkeypatch.setattr(subject,"PAIR_ROW_BYTES",100)
    owner,capture,calls,results=_paired_fixture(monkeypatch)
    assert owner.paired.incomplete and owner.paired.state is None
    assert calls["detect"]==1 and results[0][0].supports.dtype==subject.region_growth.np.float32


def _paired_six_frames(monkeypatch):
    import weakref
    owner=subject.ReadingAttritionObserver(paired_segment_support=True)
    refs=[];begin=owner.paired.begin
    def retain_weakrefs(*args):
        result=begin(*args)
        refs.extend(weakref.ref(owner.paired.state[key]) for key in ("raster","workspace"))
        return result
    monkeypatch.setattr(owner.paired,"begin",retain_weakrefs)
    for index in range(6):
        _,capture,calls,_=_paired_fixture(monkeypatch,observer=owner,frame=f"image-{index:04d}")
        assert calls["detect"]==1 and owner.paired.state is None
        assert all(ref() is None for ref in refs)
    capture.graph=SimpleNamespace(frame_ids=tuple(f"image-{index:04d}" for index in range(6)))
    return owner,capture,refs


def test_paired_six_frames_36_rows_reconcile_and_release_transients(monkeypatch):
    owner,capture,refs=_paired_six_frames(monkeypatch)
    record,attrition=_paired_record(owner,capture)
    assert record["status"]=="complete" and all(record["checks"].values())
    assert len(record["frame_rows"])==6 and len(record["rows"])==36
    assert record["decision"]=="continue_candidate" and record["qualifying_targets"]==[["image-0000",0,-1]]
    assert owner.paired.items==42 and all(ref() is None for ref in refs)
    assert len(subject.canonical_bytes(record))<subject.PAIR_BYTES
    assert all(len(subject.canonical_bytes(row))<=2048 for row in record["rows"]+record["frame_rows"])
    def scalar_tree(value):
        if type(value) is dict: return all(scalar_tree(v) for v in value.values())
        if type(value) in (list,tuple): return all(scalar_tree(v) for v in value)
        return type(value) in (str,int,float,bool,type(None))
    assert scalar_tree(owner.paired.frames)
    assert subject._pair_account(record,attrition)==(record["checks"],"complete")


@pytest.mark.parametrize("fault",["missing","duplicate","seventh","nested","omitted-capture"])
def test_paired_frame_census_refuses_incomplete_or_extra_frames(monkeypatch,fault):
    owner,capture,_=_paired_six_frames(monkeypatch);pair=owner.paired
    if fault=="missing": del pair.frames["image-0005"]
    elif fault=="omitted-capture": capture.graph=SimpleNamespace(frame_ids=capture.graph.frame_ids[:-1])
    else:
        with pytest.raises(ValueError):
            if fault=="duplicate": pair.enter_frame("image-0000")
            if fault=="seventh": pair.enter_frame("image-0006")
            if fault=="nested":
                pair.frames={};pair.enter_frame("image-0000");pair.enter_frame("image-0001")
    record,_=_paired_record(owner,capture)
    assert record["status"]=="incomplete" and record["decision"]=="inconclusive"
    assert not record["qualifying_targets"] and pair.state is None


@pytest.mark.parametrize("fault",["labels","grey","usable","calibration","config","scale", "graph-labels","graph-usable","graph-support","graph-lab","graph-nodes"])
def test_paired_original_graph_input_identity_mismatch_cleans_up(monkeypatch,fault):
    owner=subject.ReadingAttritionObserver(paired_segment_support=True)
    original=owner.paired.graph
    def corrupt(result,args,kwargs):
        graph=result[0];args=list(args);kwargs=dict(kwargs)
        if fault in ("labels","grey","usable"):
            index={"labels":0,"grey":1,"usable":3}[fault];args[index]=args[index].copy()
        elif fault in ("calibration","config"):
            index=4 if fault=="calibration" else 5;args[index]=dataclasses.replace(args[index])
        elif fault=="scale": kwargs["original_pixel_scale"]=(2.,1.)
        elif fault=="graph-nodes": graph=dataclasses.replace(graph,node_area=graph.node_area[:-1])
        else:
            field={"graph-labels":"labels","graph-usable":"usable","graph-support":"supports","graph-lab":"lab"}[fault]
            graph=dataclasses.replace(graph,**{field:getattr(graph,field).copy()})
        return original((graph,result[1]),tuple(args),kwargs)
    monkeypatch.setattr(owner.paired,"graph",corrupt)
    owner,_,calls,_=_paired_fixture(monkeypatch,observer=owner)
    assert owner.paired.incomplete and owner.paired.state is None and calls["detect"]==1


@pytest.mark.parametrize("flag",["omitted",None,False,1])
def test_paired_evaluator_default_omitted_none_nonliteral_isolation(monkeypatch,flag):
    import numpy as np
    from test_region_growth import _raster_controls,_calibration
    rg=subject.region_growth
    calls=_raster_controls(monkeypatch,[(0,0,80,0)])
    labels=np.zeros((2,81),np.int32);grey=np.zeros(labels.shape,np.uint8)
    usable=np.ones(labels.shape,bool);lab=np.zeros((*labels.shape,3),np.float32)
    args=(labels,grey,lab,usable,_calibration(),rg.DEFAULT_REGION_GROWTH_CONFIG)
    baseline=rg._build_graph(*args)
    monkeypatch.setattr(subject,"PairedSupportObserver",lambda *a:pytest.fail("nonliteral observer construction"))
    monkeypatch.setattr(rg,"_raster_allocate_segment",lambda *a:pytest.fail("nonliteral raster allocation"))
    owner=subject.ReadingAttritionObserver(**({} if flag=="omitted" else dict(paired_segment_support=flag)))
    assert owner.paired is None
    def measure(frame):
        with rg._scalar_attrition_context(owner,frame=frame.frame_id): return rg._build_graph(*args)
    monkeypatch.setattr(subject.lattice_measure,"_measure_frame",measure)
    original_support=rg._segment_support
    result=subject.GraphCapture(owner).invoke(lambda:subject.lattice_measure._measure_frame(SimpleNamespace(frame_id="image-0000")))
    assert rg._segment_support is original_support and result[1]==baseline[1]
    for field in ("supports","plane_scores","sides","node_area"):
        np.testing.assert_array_equal(getattr(result[0],field),getattr(baseline[0],field))
    assert result[0].supports.dtype==baseline[0].supports.dtype and calls["detect"]==2


def test_paired_project_actual_losses_target_transfer_and_overlapping_union():
    import numpy as np
    rg=subject.region_growth
    old=np.asarray([[0.,48.,48.],[0.,48.,48.],[0.,0.,0.]],np.float32)
    new=np.asarray([[48.,0.,48.],[0.,0.,0.],[0.,48.,48.]],np.float32)
    graph=SimpleNamespace(supports=old,plane_scores=rg._normal_scores(old),sides=np.ones((3,3),np.int8))
    rows,union,_=subject._pair_project("image-0000",graph,new,rg.DEFAULT_REGION_GROWTH_CONFIG)
    # Transfer target is axis2: choose an old axis1 atom and a new axis2 atom.
    transfer_old=np.asarray([[48.,0.,48.]],np.float32);transfer_new=np.asarray([[48.,48.,0.]],np.float32)
    transfer_graph=SimpleNamespace(supports=transfer_old,plane_scores=rg._normal_scores(transfer_old),sides=np.ones((1,3),np.int8))
    transfer_rows,transfer_union,_=subject._pair_project("image-0000",transfer_graph,transfer_new,rg.DEFAULT_REGION_GROWTH_CONFIG)
    transferred=next(row for row in transfer_rows if row["axis"]==2 and row["side"]==1)
    assert transferred["transferred"]==1 and transferred["globally_new"]==0 and transferred["witness"] is not None
    assert transfer_union==dict(both=1,old_only=0,new_only=0,neither=0)
    assert any(row["partition"]["old_only"]==2 and row["coverage"]=="positive_to_positive" for row in rows)
    assert any(row["partition"]["old_only"]==1 and row["coverage"]=="positive_to_zero" for row in transfer_rows)
    assert union==dict(both=1,old_only=1,new_only=1,neither=0)
    # Identical balanced supports with zero thresholds genuinely overlap axes;
    # projection must OR masks, not sum the three positive-side marginals.
    balanced=np.full((3,3),48.,np.float32)
    overlap_graph=SimpleNamespace(supports=balanced,plane_scores=rg._normal_scores(balanced),sides=np.ones((3,3),np.int8))
    config=dataclasses.replace(rg.DEFAULT_REGION_GROWTH_CONFIG,min_seed_score=0.,min_seed_margin=0.)
    overlap_rows,overlap_union,_=subject._pair_project("image-0001",overlap_graph,balanced.copy(),config)
    assert sum(row["old"]["eligible"] for row in overlap_rows)==9
    assert overlap_union==dict(both=3,old_only=0,new_only=0,neither=0)


def test_paired_transient_cap_refuses_before_allocator_and_projection(monkeypatch):
    import numpy as np
    monkeypatch.setattr(subject,"PAIR_TEMP_BYTES",1)
    monkeypatch.setattr(subject.region_growth,"_RasterSupportWorkspace",lambda:pytest.fail("allocation before preflight"))
    with pytest.raises(ValueError,match="transient budget"):
        _paired_direct_begin(monkeypatch)
    graph=SimpleNamespace(supports=np.zeros((1,3),np.float32),plane_scores=np.zeros((1,3),np.float32),
                          sides=np.ones((1,3),np.int8))
    with pytest.raises(ValueError,match="transient budget"):
        subject._pair_project("image-0000",graph,np.zeros((1,3),np.float32),subject.region_growth.DEFAULT_REGION_GROWTH_CONFIG)


def test_paired_zero_retained_census_is_measured_without_fabricated_mass(monkeypatch):
    owner,capture,_,_=_paired_fixture(monkeypatch,case="empty")
    record,_=_paired_record(owner,capture)
    frame=record["frame_rows"][0]
    assert {k:v for k,v in frame["census"].items() if k!="evidence"}==dict(retained=0,observed=0,zero=0,visits=0,nodes=2)
    for field,value in frame["mass"].items():
        assert value==([1e-8]*3 if field=="allowance" else [0.]*3)
    assert record["decision"]=="stop_no_new_only" and all(row["witness"] is None for row in record["rows"])


def test_paired_max_scattered_workload_peak_includes_projection_temporaries(monkeypatch):
    import time
    import tracemalloc
    import weakref
    import numpy as np
    from test_region_growth import _raster_controls,_calibration
    rg=subject.region_growth;nodes=20000;shape=(960,960)
    yy,xx=np.indices(shape,dtype=np.int32)
    labels=((xx*17+yy*1031)%nodes).astype(np.int32)
    assert labels.max()+1==nodes
    grey=np.zeros(shape,np.uint8);usable=np.ones(shape,bool);lab=np.zeros((*shape,3),np.float32)
    rows=(np.arange(5000)*37)%960
    lines=np.column_stack((np.zeros(5000),rows,np.full(5000,959),rows)).astype(float)
    residuals=np.full((5000,3),100.);residuals[np.arange(5000),np.arange(5000)%3]=0.
    calls=_raster_controls(monkeypatch,lines,residuals=residuals)
    # Upstream atomization/adjacency are authored controls. Original support,
    # score/sides producer, graph wrapper and paired projection still execute.
    xy=np.column_stack((np.arange(nodes)%960,np.arange(nodes)//960)).astype(np.float32)
    monkeypatch.setattr(rg,"_node_features",lambda *args:(np.ones(nodes,np.int32),np.zeros((nodes,3),np.float32),np.zeros(nodes,np.float32),xy))
    monkeypatch.setattr(rg,"_graph_adjacency",lambda *args:((),)*nodes)
    owner=subject.ReadingAttritionObserver(paired_segment_support=True)
    begin=owner.paired.begin;refs=[]
    def start_measurement(*args):
        # Start after shared source inputs/evidence/support were allocated.
        tracemalloc.start();result=begin(*args)
        refs.extend(weakref.ref(owner.paired.state[key]) for key in ("workspace","raster"))
        return result
    monkeypatch.setattr(owner.paired,"begin",start_measurement)
    config=rg.DEFAULT_REGION_GROWTH_CONFIG;cal=_calibration(width=960,height=960)
    original_support=rg._segment_support;original_build=rg._build_graph;result=[]
    def measure(frame):
        with rg._scalar_attrition_context(owner,frame=frame.frame_id):
            graph=rg._build_graph(labels,grey,lab,usable,cal,config)[0];result.append(graph)
            for axis in range(3):
                for side in (-1,1): rg._seed_and_grow(graph,axis,side,config)
        return graph
    monkeypatch.setattr(subject.lattice_measure,"_measure_frame",measure)
    capture=subject.GraphCapture(owner);started=time.perf_counter()
    try:
        graph=capture.invoke(lambda:subject.lattice_measure._measure_frame(SimpleNamespace(frame_id="image-0000")))
        elapsed=time.perf_counter()-started
        _,peak=tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert rg._segment_support is original_support and rg._build_graph is original_build
    assert graph is result[0] and not owner.paired.incomplete, {
        'paired_seconds':owner.paired.total_seconds,
        'states':[(entry['state'],entry['reason']) for entry in owner.paired.frames.values()]}
    assert owner.paired.state is None and all(ref() is None for ref in refs)
    frame=owner.paired.frames["image-0000"]
    assert frame["census"]["retained"]==frame["census"]["observed"]==5000
    assert frame["census"]["visits"]==4_800_000 and frame["census"]["nodes"]==20_000
    assert calls==dict(enhance=1,detect=1,clip=1,residual=1,weight=1)
    assert owner.paired.peak_numeric_bytes==296_640
    assert owner.paired.accounted_transient_bytes==4_027_712<=subject.PAIR_TEMP_BYTES
    assert peak<=owner.paired.accounted_transient_bytes
    capture.graph=SimpleNamespace(frame_ids=("image-0000",))
    capture.arguments=((),dict(lattice_result=SimpleNamespace(readings=())))
    record,_=_paired_record(owner,capture)
    assert record["status"]=="complete"
    print(f"paired max-scattered: segments=5000 side=960 frame=960x960 nodes=20000 visits=4800000 "
          f"elapsed_seconds={elapsed:.6f} paired_seconds={owner.paired.total_seconds:.6f} "
          f"traced_peak_bytes={peak} accounted_transient_bytes={owner.paired.accounted_transient_bytes} "
          f"retained_numeric_peak_bytes={owner.paired.peak_numeric_bytes}")


def test_unit_weight_exact_sparse_ids_denominator_conservation_and_mask_isolation(monkeypatch):
    import numpy as np
    rg=subject.region_growth
    owner=subject.ReadingAttritionObserver(unit_weight_sparse_support=True)
    owner,values=_paired_direct_begin(monkeypatch,labels=np.pad(np.ones((2,1),np.int32),((0,0),(0,80))),owner=owner)
    pair=owner.paired
    assert pair.state['workspace'] is None
    # Amend only authored evidence, before begin fingerprinting, then begin anew.
    pair.discard('authored_reset')
    owner=subject.ReadingAttritionObserver(unit_weight_sparse_support=True);pair=owner.paired
    pair.enter_frame('image-0000')
    values[1][:]=False
    values[7][0]=.25;values[8][0]=[.5,.5,0.]
    originals=[v.copy() for v in values if isinstance(v,np.ndarray)]
    ids=np.asarray([-1,0,0,1],np.int32);ids_before=ids.copy()
    actual=np.asarray([[5.,5.,0.],[2.5,2.5,0.]],np.float32)
    with rg._scalar_attrition_context(owner,frame='image-0000'):
        pair.begin(*values)
        pair.segment(0,*values[5:9],ids,4)
        pair.segment(1,*values[5:9],None,None)
        np.testing.assert_array_equal(pair.state['raster'],[[20.,20.,0.],[10.,10.,0.]])
        pair.finalize(actual);pair.returned((actual,{}))
    mass=pair.state['mass']
    assert mass['potential']==[10.,10.,0.]
    assert mass['counterfactual_potential']==[40.,40.,0.]
    assert mass['old_unsupported']==[2.5,2.5,0.]
    assert mass['new_unsupported']==[10.,10.,0.]
    assert mass['old']==[7.5,7.5,0.] and mass['new']==[30.,30.,0.]
    np.testing.assert_array_equal(ids,ids_before)
    for before,after in zip(originals,[v for v in values if isinstance(v,np.ndarray)]):
        np.testing.assert_array_equal(before,after)
    pair.discard('test_cleanup');assert pair.state is None


@pytest.mark.parametrize('enabled',[None,False,1,'true'])
def test_unit_weight_literal_opt_in_and_default_omission(enabled):
    assert subject.ReadingAttritionObserver().paired is None
    assert subject.ReadingAttritionObserver(unit_weight_sparse_support=enabled).paired is None
    with pytest.raises(ValueError,match='separate'):
        subject.ReadingAttritionObserver(paired_segment_support=True,unit_weight_sparse_support=True)


@pytest.mark.parametrize('weights',[[1.,1.],[.01,.01],[0.,0.]])
def test_unit_weight_one_source_pass_weight_only_target_and_default_noninterference(monkeypatch,weights):
    import numpy as np
    owner=subject.ReadingAttritionObserver(unit_weight_sparse_support=True)
    def forbidden(*args,**kwargs): raise AssertionError('dense allocator entered unit mode')
    monkeypatch.setattr(subject.region_growth,'_raster_allocate_segment',forbidden)
    monkeypatch.setattr(subject.region_growth,'_RasterSupportWorkspace',forbidden)
    owner,capture,calls,results=_paired_fixture(monkeypatch,observer=owner,weights=weights,yaw=0)
    record,attrition=_paired_record(owner,capture)
    assert record['schema']==subject.UNIT_WEIGHT_SCHEMA and record['causal_scope']==subject.UNIT_WEIGHT_SCOPE
    assert record['status']=='complete' and all(record['checks'].values())
    assert subject._pair_account(record,attrition,unit_weight_sparse=True)==(record['checks'],'complete')
    assert calls==dict(enhance=1,detect=1,clip=1,residual=1,weight=1)
    frame=record['frame_rows'][0]
    assert frame['mass']['counterfactual_potential']==[0.,80.,80.]
    assert 'evidence' not in frame['census'] and 'visits' not in frame['census']
    np.testing.assert_allclose(results[0][0].supports.sum(axis=0),[0.,80.*weights[1],80.*weights[0]],rtol=2e-7)
    if weights==[1.,1.]:
        assert record['decision']=='stop_no_new_only'
        assert all(row['old']==row['new'] for row in record['rows'])
    else:
        assert record['decision']=='continue_candidate'
        assert ['image-0000',0,-1] in record['qualifying_targets']
        witness=next(r['witness'] for r in record['rows'] if r['witness'])
        assert all(witness['new']['passes']) and not all(witness['old']['passes'])
    encoded=subject.canonical_bytes(record)
    assert b'node_id' not in encoded and b'coordinates' not in encoded and b'labels' not in encoded
    assert owner.paired.state is None


def test_unit_weight_sparse_spatial_separation_cannot_create_target(monkeypatch):
    owner=subject.ReadingAttritionObserver(unit_weight_sparse_support=True)
    owner,capture,_,_=_paired_fixture(monkeypatch,observer=owner,weights=[.01,.01])
    record,_=_paired_record(owner,capture)
    target=next(r for r in record['rows'] if r['axis']==0 and r['side']==-1)
    assert target['new']['eligible']==0 and not record['qualifying_targets']
    assert record['decision'] in ('stop_non_target_only','stop_no_new_only')


def test_unit_weight_actual_cross_row_losses_are_reported(monkeypatch):
    owner=subject.ReadingAttritionObserver(unit_weight_sparse_support=True)
    lines=[(8,32,88,32)]*100+[(20,8,20,88)]
    owner,capture,_,_=_paired_fixture(monkeypatch,case='single',observer=owner,lines=lines,
        residuals=[[0,100,100]]*100+[[100,0,100]],weights=[.01]*100+[1.])
    record,attrition=_paired_record(owner,capture)
    assert record['status']=='complete'
    losses=[r for r in record['rows'] if r['partition']['old_only']]
    assert losses and record['losses']
    assert sum(r['partition']['old_only'] for r in losses)>=1
    assert record['frame_rows'][0]['union']['old_only']==1
    assert subject._pair_account(record,attrition,unit_weight_sparse=True)==(record['checks'],'complete')


@pytest.mark.parametrize('fault',['samples','new-mass','raw-mass','allowance','axis-bool','scope','mixed'])
def test_unit_weight_strict_resealed_report_validation(monkeypatch,fault):
    owner=subject.ReadingAttritionObserver(unit_weight_sparse_support=True)
    owner,capture,_,_=_paired_fixture(monkeypatch,observer=owner,weights=[.01,.01],yaw=0)
    record,attrition=_paired_record(owner,capture)
    frame=record['frame_rows'][0]
    if fault=='samples': frame['census']['sparse_samples']=640001
    if fault=='new-mass': frame['mass']['new'][0]+=1
    if fault=='raw-mass': frame['mass']['counterfactual_potential'][1]+=1
    if fault=='allowance': frame['mass']['counterfactual_allowance'][1]+=1
    if fault=='axis-bool': record['rows'][0]['axis']=False
    if fault=='scope': record['causal_scope']='combined_centerline_and_usable_mask_effect_only'
    base=dict(schema=subject.SCHEMA,producer=subject.PRODUCER,status='incomplete',reason='upstream',field='',
              g_calls=0,graph=None,run={},notes=[],reading_attrition=attrition,unit_weight_sparse_support=record)
    if fault=='mixed': base['paired_segment_support']=record
    with pytest.raises(subject.DiagnosticError): subject.validate_report(subject._seal(base))


def test_unit_weight_not_reached_source_seed_inconclusive_and_valid_report(monkeypatch):
    owner=subject.ReadingAttritionObserver(unit_weight_sparse_support=True)
    owner,capture,_,_=_paired_fixture(monkeypatch,observer=owner,weights=[.01,.01],yaw=0,emit_seeds=False)
    record,attrition=_paired_record(owner,capture)
    assert record['status']=='incomplete' and record['decision']=='inconclusive'
    assert not record['checks']['source_seed_join'] and not record['qualifying_targets']
    assert all(r['state']=='measured' for r in record['rows'])
    base=dict(schema=subject.SCHEMA,producer=subject.PRODUCER,status='incomplete',
              reason='unit_weight_sparse_support_incomplete',field='observer',g_calls=0,
              graph=None,run={},notes=[],reading_attrition=attrition,unit_weight_sparse_support=record)
    subject.validate_report(subject._seal(base))


@pytest.mark.parametrize('case',['empty','mask-hole'])
def test_unit_weight_zero_census_mask_independence_and_report_cap(monkeypatch,case):
    owner=subject.ReadingAttritionObserver(unit_weight_sparse_support=True)
    owner,capture,_,_=_paired_fixture(monkeypatch,case=case,observer=owner)
    record,_=_paired_record(owner,capture)
    assert record['status']=='complete'
    assert record['frame_rows'][0]['mass']['new_unsupported']==[0.,0.,0.]
    if case=='empty':
        assert record['frame_rows'][0]['census']['sparse_samples']==0
        assert record['decision']=='stop_no_new_only'
    owner.max_bytes=1
    with pytest.raises(ValueError,match='combined envelope'):
        _paired_record(owner,capture)


@pytest.mark.parametrize('fault',['float-id','out-of-range','wrong-denominator','nonfinite-weight'])
def test_unit_weight_bad_sparse_evidence_refuses_and_cleans_up(monkeypatch,fault):
    import numpy as np
    owner=subject.ReadingAttritionObserver(unit_weight_sparse_support=True)
    owner,values=_paired_direct_begin(monkeypatch,owner=owner)
    ids=np.zeros(4,np.int32);denominator=4
    if fault=='float-id': ids=ids.astype(float)
    if fault=='out-of-range': ids[0]=1
    if fault=='wrong-denominator': denominator=3
    if fault=='nonfinite-weight': values[7][0]=np.nan
    with subject.region_growth._scalar_attrition_context(owner,frame='image-0000'):
        with pytest.raises(ValueError): owner.paired.segment(0,*values[5:9],ids,denominator)
    assert owner.paired.incomplete and owner.paired.state is None


def test_unit_weight_actual_source_support_matches_omitted_observer(monkeypatch):
    import numpy as np
    from test_region_growth import _raster_controls, _calibration
    owner=subject.ReadingAttritionObserver(unit_weight_sparse_support=True)
    owner,capture,_,results=_paired_fixture(monkeypatch,observer=owner,weights=[.25,.5])
    actual=results[0][0]
    _raster_controls(monkeypatch,[(8,32,88,32),(20,8,20,88)],
                     residuals=[[100,100,0],[100,0,100]],weights=[.25,.5])
    # The ordinary source call has no observer and must return the same matrix.
    support,_=subject.region_growth._segment_support(np.zeros(actual.labels.shape,np.uint8),
        actual.usable,actual.labels,_calibration(yaw=180,width=97,height=97),
        subject.region_growth.DEFAULT_REGION_GROWTH_CONFIG)
    np.testing.assert_array_equal(actual.supports,support)


@pytest.mark.parametrize('removed',[None,1,2])
def test_unit_weight_required_tangent_family_cannot_be_invented_when_colocated(monkeypatch,removed):
    owner=subject.ReadingAttritionObserver(unit_weight_sparse_support=True)
    residuals=[[100,100,0],[100,0,100]]  # Axis0 plane needs these axis2/axis1 tangents.
    if removed==2: residuals[0]=[100,100,100]
    if removed==1: residuals[1]=[100,100,100]
    owner,capture,_,_=_paired_fixture(monkeypatch,observer=owner,
        weights=[.01,.01],yaw=0,residuals=residuals)
    record,_=_paired_record(owner,capture)
    target=next(r for r in record['rows'] if r['axis']==0 and r['side']==-1)
    assert target['denominator']==1  # Both authored lines remain on this same atom.
    assert target['old']['eligible']==0
    mass=record['frame_rows'][0]['mass']['counterfactual_potential']
    if removed is None:
        assert mass==[0.,80.,80.]
        assert target['new']['eligible']==1 and target['new']['cooccurrence']==1
        assert ['image-0000',0,-1] in record['qualifying_targets']
        assert all(target['witness']['new']['passes'])
    else:
        other=3-removed
        assert mass[removed]==0. and mass[other]==80.
        assert target['new']['positive'][removed]==0 and target['new']['positive'][other]==1
        assert target['new']['eligible']==0 and target['new']['cooccurrence']==0
        assert target['witness'] is None and not record['qualifying_targets']
        assert record['decision']=='stop_no_new_only'


@pytest.mark.parametrize('case,raw',[('single',2717.),('empty',1e-12)])
def test_unit_weight_coherently_resealed_impossible_raw_potential_refused(monkeypatch,case,raw):
    owner=subject.ReadingAttritionObserver(unit_weight_sparse_support=True)
    owner,capture,_,_=_paired_fixture(monkeypatch,case=case,observer=owner)
    record,attrition=_paired_record(owner,capture)
    base=dict(schema=subject.SCHEMA,producer=subject.PRODUCER,status='incomplete',reason='upstream',field='',
        g_calls=0,graph=None,run={},notes=[],reading_attrition=attrition,unit_weight_sparse_support=record)
    subject.validate_report(subject._seal(deepcopy(base)))  # Authored original is valid.
    mass=record['frame_rows'][0]['mass'];axis=0
    assert mass['counterfactual_potential'][axis]==0.
    assert (record['frame_rows'][0]['census']['retained']-record['frame_rows'][0]['census']['zero'])==(2 if case=='single' else 0)
    mass['counterfactual_potential'][axis]=raw
    mass['new'][axis]=raw;mass['new_ideal'][axis]=raw
    eps=2**-23
    mass['counterfactual_allowance'][axis]=(5000*eps/(1-5000*eps)+2*eps)*raw+1e-8
    # Conservation and self-derived float32 allowance remain exactly coherent.
    assert mass['new'][axis]+mass['new_unsupported'][axis]==raw
    assert mass['new_ideal'][axis]+mass['new_unsupported'][axis]==raw
    resealed=subject._seal(base)
    with pytest.raises(ValueError,match='unit sparse raw potential bound'):
        subject._pair_account(record,attrition,unit_weight_sparse=True)
    with pytest.raises(subject.DiagnosticError) as caught:
        subject.validate_report(resealed)
    assert str(caught.value.__cause__)=='unit sparse raw potential bound'


@pytest.mark.parametrize('policy',['omitted','structural','unit'])
def test_actual_support_policy_harness_reaches_real_proposal_allocator_and_seeds(monkeypatch,policy):
    import numpy as np
    from test_region_growth import _line_weight_policy_target_graph, _calibration
    rg=subject.region_growth
    authored,_,_,calls=_line_weight_policy_target_graph(monkeypatch)
    monkeypatch.setattr(rg,'_local_lab_xy_atoms',lambda *args,**kwargs:
        (authored.labels.copy(),dict(connectivity_mask_preserved=True)))
    saved_build,saved_measure=rg._build_graph,subject.lattice_measure._measure_frame
    options={} if policy=='omitted' else dict(support_line_weight_policy=policy)
    capture=subject.GraphCapture(**options)
    before_weight=calls['weight']
    image=np.zeros((*authored.labels.shape,3),np.uint8)
    usable=np.ones(authored.labels.shape,bool)
    def delivery():
        # Real proposal keeps its ordinary default; the diagnostic graph seam
        # injects only the explicit private allocator policy downstream.
        return rg.propose_image_planes_v2(image,usable,_calibration(yaw=180,width=97,height=97))
    result=capture.invoke(delivery)
    effective='structural' if policy=='omitted' else policy
    assert capture.support_line_weight_policy==effective
    assert calls['weight']-before_weight==int(effective=='structural')
    assert any(p.axis==0 and p.side==-1 for p in result.proposals)==(effective=='unit')
    assert rg._build_graph is saved_build and subject.lattice_measure._measure_frame is saved_measure
    # Sealed reports state the effective experiment even on upstream/g-call refusal.
    report=subject.build_report(capture,{})
    assert report['support_line_weight_policy']==effective
    subject.validate_report(report)
    changed=deepcopy(report);changed['support_line_weight_policy']='unit' if effective=='structural' else 'structural'
    with pytest.raises(subject.DiagnosticError,match='digest'): subject.validate_report(changed)
    # Default after an opt-in is structurally identical and no setting leaks.
    subsequent=subject.GraphCapture().invoke(delivery)
    assert not subsequent.proposals and subsequent.reason=='insufficient_plane_seeds'
    assert rg._build_graph is saved_build


def test_actual_support_policy_existing_paired_observers_remain_complete(monkeypatch):
    owner=subject.ReadingAttritionObserver(paired_segment_support=True)
    owner,capture,calls,_=_paired_fixture(monkeypatch,observer=owner,weights=[.25,.5],
        support_line_weight_policy='unit')
    record,attrition=_paired_record(owner,capture)
    assert capture.support_line_weight_policy=='unit'
    assert calls['weight']==0
    assert record['status']==attrition['status']=='complete'
    assert all(record['checks'].values()) and owner.paired.state is None
    assert record['frame_rows'][0]['mass']['potential']==[0.,80.,80.]


@pytest.mark.parametrize('policy',[None,False,True,1,'UNIT','unknown',''])
def test_actual_support_policy_invalid_setting_refuses_before_input_or_delivery(policy):
    def images():
        pytest.fail('invalid policy consumed input');yield None
    with pytest.raises(subject.DiagnosticError,match='support_line_weight_policy'):
        subject.run_once(images(),project='unused',destination='unused',diagnostic='unused',
            name='unused',description='unused',support_line_weight_policy=policy)
    with pytest.raises(subject.DiagnosticError,match='support_line_weight_policy'):
        subject.GraphCapture(support_line_weight_policy=policy)


def test_actual_support_policy_unit_counterfactual_combination_refuses_before_inputs():
    def images():
        pytest.fail('ambiguous mode consumed input');yield None
    with pytest.raises(subject.DiagnosticError,match='unit_policy_counterfactual_combination'):
        subject.run_once(images(),project='unused',destination='unused',diagnostic='unused',
            name='unused',description='unused',reading_attrition=True,
            unit_weight_sparse_support=True,support_line_weight_policy='unit')
    owner=subject.ReadingAttritionObserver(unit_weight_sparse_support=True)
    with pytest.raises(subject.DiagnosticError,match='unit_policy_counterfactual_combination'):
        subject.GraphCapture(owner,support_line_weight_policy='unit')
    assert subject.GraphCapture(owner).support_line_weight_policy=='structural'


@pytest.mark.parametrize('policy',[None,False,1,'invalid'])
def test_actual_support_policy_resealed_invalid_provenance_refuses(policy):
    report=subject.build_report(subject.GraphCapture(),{})
    report['support_line_weight_policy']=policy
    report.pop('result_sha256');subject._seal(report)
    with pytest.raises(subject.DiagnosticError,match='support_line_weight_policy'):
        subject.validate_report(report)


def test_actual_support_policy_legacy_report_omission_remains_loadable_byte_preserving(tmp_path):
    report=subject.build_report(subject.GraphCapture(),{})
    report.pop('support_line_weight_policy');report.pop('result_sha256');subject._seal(report)
    original=subject.canonical_bytes(report)
    path=tmp_path/'legacy-report.json';path.write_bytes(original)
    loaded=subject.validate_report(json.loads(path.read_bytes()))
    assert 'support_line_weight_policy' not in loaded
    assert subject.canonical_bytes(loaded)==original and path.read_bytes()==original


@pytest.mark.parametrize('exception',[RuntimeError('source failure'),KeyboardInterrupt()])
def test_actual_support_policy_wrapper_exception_cleanup(monkeypatch,exception):
    rg=subject.region_growth
    seen=[]
    def original(*args,**kwargs):
        seen.append(kwargs['_support_line_weight_policy']);raise exception
    monkeypatch.setattr(rg,'_build_graph',original)
    capture=subject.GraphCapture(support_line_weight_policy='unit')
    with pytest.raises(type(exception)) as caught:
        capture.invoke(lambda:rg._build_graph())
    assert caught.value is exception and seen==['unit']
    assert rg._build_graph is original


def test_actual_support_policy_unit_refuses_foreign_invalid_policy(monkeypatch):
    rg=subject.region_growth
    def forbidden(*args,**kwargs): pytest.fail('foreign policy reached source')
    monkeypatch.setattr(rg,'_build_graph',forbidden)
    capture=subject.GraphCapture(support_line_weight_policy='unit')
    with pytest.raises(subject.DiagnosticError,match='support_line_weight_policy'):
        capture.invoke(lambda:rg._build_graph(_support_line_weight_policy='invalid'))
    assert rg._build_graph is forbidden


@pytest.mark.parametrize('policy',['omitted','structural','unit'])
def test_actual_support_policy_cli_choice_and_forwarding_without_run(monkeypatch,policy):
    calls=[]
    def fake(*args,**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(report={'status':'incomplete','reason':'synthetic','result_sha256':'a'*64,'run':{}},
            diagnostic_error=None,production_error=None)
    monkeypatch.setattr(subject,'run_once',fake)
    argv=['--image','unused.png','--project','unused.mcscan','--destination','unused.litematic',
          '--diagnostic','unused.json','--name','synthetic','--description','synthetic']
    if policy!='omitted': argv+=['--support-line-weight-policy',policy]
    assert subject.main(argv)==1  # Existing incomplete-report CLI disposition.
    assert calls[0]['support_line_weight_policy']==('structural' if policy=='omitted' else policy)


def test_actual_support_policy_cli_invalid_choice_refuses_without_run(monkeypatch):
    monkeypatch.setattr(subject,'run_once',lambda *args,**kwargs:pytest.fail('invalid CLI reached run'))
    argv=['--image','unused.png','--project','unused.mcscan','--destination','unused.litematic',
          '--diagnostic','unused.json','--name','synthetic','--description','synthetic',
          '--support-line-weight-policy','invalid']
    with pytest.raises(SystemExit) as caught: subject.main(argv)
    assert caught.value.code==2


def test_actual_support_policy_resealed_counterfactual_combination_refuses():
    report=subject.build_report(subject.GraphCapture(support_line_weight_policy='unit'),{})
    report['unit_weight_sparse_support']=None
    report.pop('result_sha256');subject._seal(report)
    with pytest.raises(subject.DiagnosticError,match='unit_policy_counterfactual_combination'):
        subject.validate_report(report)
