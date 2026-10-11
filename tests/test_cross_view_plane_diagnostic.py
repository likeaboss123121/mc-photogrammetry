"""Offline collector, cap, selection, cleanup and no-file entry controls."""
from collections import Counter
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import weakref

from PIL import Image
import numpy as np
import pytest

from scripts.evaluation import cross_view_plane_diagnostic as diagnostic
from test_cross_view_plane_continuity import _scene, _snapshot, _matches


def _capture(*, folded=False):
    observation, snapshot = _scene(folded=folded)
    watch = diagnostic.Watch(clock=lambda: 0., rss=lambda: 1024)
    capture = diagnostic.Capture(watch)
    capture.pair(observation)
    frames = []
    for name in (observation.left_id, observation.right_id, "third", "fourth", "fifth", "sixth"):
        pixel = observation.right_pixel_sha256 if name == observation.right_id else observation.left_pixel_sha256
        shape = observation.right_shape if name == observation.right_id else observation.left_shape
        frames.append(SimpleNamespace(frame_id=name, image_sha256=pixel, image_shape=shape))
        native_digest=diagnostic.rg._diagnostic_native_mask_digest(name,shape,snapshot.native_usable_bytes)
        capture.atom(_snapshot(snapshot, frame_id=name, pixel_sha256=pixel, source_shape=shape,
                               native_usable_sha256=native_digest))
    graph = SimpleNamespace(status="ok", evidence=(SimpleNamespace(
        left_id=observation.left_id, right_id=observation.right_id,
        evidence_sha256=observation.evidence_sha256, matches_sha256=observation.matches_sha256,
        match_count=observation.source_count, status=observation.status, reason=observation.reason,
        parallax_degrees=observation.parallax_degrees,ambiguity_margin=observation.ambiguity_margin),))
    entry = SimpleNamespace(session=SimpleNamespace(frames=tuple(frames),
        session_sha256=observation.session_sha256), graph=graph)
    return capture, entry


@pytest.mark.parametrize("folded,status,reason", [
    (False,"appearance_only","training_frozen_bilateral_holdout_passed"),
    (True,"abstain","no_training_projective_model")])
def test_cross_view_runner_known_plane_perpendicular_fold_scalar_counts(folded,status,reason):
    capture, entry = _capture(folded=folded)
    result = diagnostic.summarize(capture,entry)
    assert result["status"] == "complete"
    assert result["unions_evaluated"] == 1
    assert result["correspondences_evaluated"] == 16
    assert result["hypotheses_evaluated"] == 36
    assert result["status_counts"] == {status:1}
    assert result["reason_counts"] == {reason:1}
    assert result["seed_authority"] is False and result["reconstruction_gain_claimed"] is False
    text = __import__("json").dumps(result)
    for forbidden in ("matches", "labels_bytes", "native_usable_bytes", "union_ids", "camera"):
        assert forbidden not in result
    assert len(text) < 16384


@pytest.mark.parametrize("kind,reason", [
    ("missing_atom","missing_or_foreign_atom_snapshot"),
    ("missing_pair","missing_or_foreign_pair_observation"),
    ("evidence","source_pair_evidence_mismatch"),
    ("right_pixel","source_pair_frame_mismatch"),
    ("count","incomplete_correspondences"),
    ("snapshot_digest","atom_snapshot_digest_mismatch"),
    ("mask_hole","correspondence_in_unusable_atom")])
def test_cross_view_runner_source_refusals_fail_closed(kind,reason):
    capture,entry=_capture()
    key=next(iter(capture.pairs));obs=capture.pairs[key]
    if kind=="missing_atom": capture.atoms.pop(obs.left_id)
    if kind=="missing_pair": capture.pairs.clear()
    if kind=="evidence": capture.pairs[key]=replace(obs,evidence_sha256="f"*64)
    if kind=="right_pixel": capture.pairs[key]=replace(obs,right_pixel_sha256="f"*64)
    if kind=="count": capture.pairs[key]=replace(obs,complete=False)
    if kind=="snapshot_digest":
        capture.atoms[obs.left_id]=replace(capture.atoms[obs.left_id],labels_bytes=bytes(len(capture.atoms[obs.left_id].labels_bytes)))
    if kind=="mask_hole":
        hole_obs,hole_snapshot=_scene(hole=True)
        capture.pairs[key]=hole_obs;capture.atoms[obs.left_id]=hole_snapshot
        entry.graph.evidence[0].evidence_sha256=hole_obs.evidence_sha256
        entry.graph.evidence[0].matches_sha256=hole_obs.matches_sha256
        entry.session.frames[0].image_sha256=hole_obs.left_pixel_sha256
    result=diagnostic.summarize(capture,entry)
    assert result["status"]=="incomplete" and result["denials"]=={reason:1}
    assert result["unions_evaluated"]==0 and result["hypotheses_evaluated"]==0


def test_cross_view_runner_legitimate_refused_pair_census_skips_models(monkeypatch):
    capture,entry=_capture();key=next(iter(capture.pairs));obs=capture.pairs[key]
    capture.pairs[key]=replace(obs,status="refused",reason="insufficient_matches")
    entry.graph.evidence[0].status="refused"
    entry.graph.evidence[0].reason="insufficient_matches"
    monkeypatch.setattr(diagnostic,"diagnose_plane_union",lambda *args:pytest.fail("refused pair modeled"))
    result=diagnostic.summarize(capture,entry)
    assert result["status"]=="complete"
    assert result["pair_status_counts"]=={"refused":1}
    assert result["refused_pair_reason_counts"]=={"insufficient_matches":1}
    assert result["hypotheses_evaluated"]==result["unions_evaluated"]==0


def test_cross_view_runner_validation_probe_never_fits_model(monkeypatch):
    capture,_=_capture()
    def forbidden(*args): raise AssertionError("validation selection attempted model fit")
    monkeypatch.setattr(__import__("scripts.evaluation.cross_view_plane_consistency",fromlist=["_fit"]),"_fit",forbidden)
    assert diagnostic._candidates(capture)==[(-16,"left","right",0,1)]


def test_cross_view_runner_deterministic_rank_and_caps(monkeypatch):
    capture,entry=_capture()
    rows=diagnostic._candidates(capture)
    assert rows==[(-16,"left","right",0,1)]
    capture.pairs=dict(reversed(list(capture.pairs.items())))
    assert diagnostic._candidates(capture)==rows
    monkeypatch.setitem(diagnostic.LIMITS,"unions_total",0)
    result=diagnostic.summarize(capture,entry)
    assert result["denials"]=={"union_total_cap":1} and result["unions_evaluated"]==0
    monkeypatch.setitem(diagnostic.LIMITS,"unions_total",128)
    monkeypatch.setitem(diagnostic.LIMITS,"unions_per_observation",0)
    capture.denials.clear()
    assert diagnostic.summarize(capture,entry)["denials"]=={"union_observation_cap":1}


def test_cross_view_runner_model_cap_prefix_never_complete():
    capture,entry=_capture()
    key=next(iter(capture.pairs));obs=capture.pairs[key]
    points=np.asarray(obs.matches)
    rows=[]
    for group in (points[:8],points[8:]):
        rows.extend(group)
        rows.extend(group+[0.,.1,0.,.1])
    modified=_matches(obs,rows)
    capture.pairs[key]=modified
    entry.graph.evidence[0].matches_sha256=modified.matches_sha256
    entry.graph.evidence[0].match_count=len(rows)
    result=diagnostic.summarize(capture,entry)
    assert result["status"]=="incomplete"
    assert result["denials"]=={"hypothesis_budget_exhausted":1}
    assert result["hypotheses_evaluated"]==128


def test_cross_view_runner_duplicate_observers_payload_and_cleanup(monkeypatch):
    capture,entry=_capture();obs=next(iter(capture.pairs.values()));snap=next(iter(capture.atoms.values()))
    capture.pair(obs);capture.atom(snap)
    assert capture.denials==Counter(duplicate_or_excess_pair=1,duplicate_or_excess_snapshot=1)
    assert diagnostic.summarize(capture,entry)["status"]=="incomplete"
    reference=weakref.ref(capture)
    capture.clear();assert not capture.pairs and not capture.atoms
    assert reference() is capture
    tiny=diagnostic.Capture(capture.watch)
    monkeypatch.setitem(diagnostic.LIMITS,"snapshot_bytes",1)
    tiny.atom(snap)
    assert not tiny.atoms and tiny.denials=={"snapshot_payload_cap":1}


@pytest.mark.parametrize("elapsed,rss,reason",[(120.,1,"wall_cap"),(0.,1024**3+1,"rss_cap")])
def test_cross_view_runner_resource_denials(elapsed,rss,reason):
    values=iter((0.,elapsed))
    watch=diagnostic.Watch(clock=lambda:next(values),rss=lambda:rss)
    with pytest.raises(diagnostic.StopDiagnostic,match=reason):watch.check()


def test_cross_view_runner_memory_persistence_replay_and_restore(monkeypatch):
    capture,_=_capture()
    from test_image_derived_connections import _simple_session
    session=_simple_session(monkeypatch,["left","right"])
    original=diagnostic.entry_mod._persist_session
    with diagnostic._memory_entry(diagnostic.inherited_input_seals()):
        assert diagnostic.entry_mod._persist_session(session=session,destination="never.created") is None
    assert diagnostic.entry_mod._persist_session is original
    with pytest.raises(RuntimeError):
        with diagnostic._memory_entry(diagnostic.inherited_input_seals()): raise RuntimeError("control")
    assert diagnostic.entry_mod._persist_session is original


def test_cross_view_runner_ingress_header_and_exact_seal_without_image_files(monkeypatch):
    image=Image.new("RGB",(3,2));buffer=io.BytesIO();image.save(buffer,format="PNG");data=buffer.getvalue()
    seals=[dict(frame=f"image-{i:04d}",path=f"synthetic-input-{i}.png",bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest().upper()) for i in range(6)]
    def discover(paths,*,recursive,ingress_sink):
        for row in seals:
            ingress_sink(str((diagnostic.REPO_ROOT/row["path"]).resolve()),SimpleNamespace(frame_id=row["frame"]),data)
        return "manifest"
    monkeypatch.setattr(diagnostic.entry_mod,"_discover",discover)
    seen=[]
    with diagnostic._memory_entry(seals):
        assert diagnostic.entry_mod._discover([],recursive=False,ingress_sink=lambda *args:seen.append(args))=="manifest"
    assert len(seen)==6
    monkeypatch.setitem(diagnostic.LIMITS,"native_pixels_per_frame",5)
    with diagnostic._memory_entry(seals):
        with pytest.raises(diagnostic.StopDiagnostic,match="native_pixel_cap"):
            diagnostic.entry_mod._discover([],recursive=False,ingress_sink=lambda *args:None)


def test_cross_view_runner_exclusive_scalar_output_and_cli_refusal(tmp_path):
    path=tmp_path/"report.json";value={"status":"complete","seed_authority":False}
    diagnostic._write_exclusive(path,value)
    original=path.read_bytes()
    with pytest.raises(FileExistsError):diagnostic._write_exclusive(path,value)
    assert path.read_bytes()==original
    with pytest.raises(SystemExit) as error:
        diagnostic.main(["--report-output",str(path),"--ingress-output",str(tmp_path/"ingress.json")])
    assert error.value.code==2
    with pytest.raises(diagnostic.StopDiagnostic,match="report_byte_cap"):
        diagnostic._write_exclusive(tmp_path/"oversize.json",{"padding":"a"*16384})
    assert not (tmp_path/"oversize.json").exists()


@pytest.mark.parametrize("status,reason,accounting,worker,expected",[
    ("terminated","wall_timeout",True,False,"wall_timeout"),
    ("terminated","aggregate_working_set_limit",True,False,"aggregate_working_set_limit"),
    ("completed","exit_zero",False,True,"supervisor_incomplete"),
    ("completed","exit_zero",True,False,"worker_report_missing"),
    ("completed","exit_zero",True,True,"appearance_diagnostic_completed")])
def test_cross_view_outer_supervisor_exclusive_scalar_finalization(tmp_path,status,reason,accounting,worker,expected):
    # Exercise the actual PS finalizer using ONLY constructed scalar fixtures;
    # this restricted branch cannot launch a monitor/worker or access images.
    packet=dict(schema="mc-cross-view-finalizer-offline-fixture-v1",task="synthetic-finalizer",inputs=[],
                project_output=str(tmp_path/"ingress.json"),json_output=str(tmp_path/"worker.json"),
                supervisor_report=str(tmp_path/"supervisor.json"))
    packet_path=tmp_path/"packet.json";packet_path.write_text(json.dumps(packet),encoding="ascii")
    monitor=dict(status=status,reason=reason,accounting_complete=accounting,root_exit_observed=True,
                 root_exit_code=0,drain_complete=True,peak_aggregate_working_set_bytes=123456,
                 monitor_elapsed_seconds=120.)
    monitor_path=tmp_path/"monitor.json";monitor_path.write_text(json.dumps(monitor),encoding="ascii")
    if worker:Path(packet["json_output"]).write_text(json.dumps(dict(status="complete",seed_authority=False)),encoding="ascii")
    args=["powershell.exe","-NoProfile","-NonInteractive","-ExecutionPolicy","Bypass","-File",
          str(diagnostic.REPO_ROOT/"scripts/evaluation/run_cross_view_plane_diagnostic.ps1"),
          "-PacketPath",str(packet_path),"-ExpectedPacketSha256",hashlib.sha256(packet_path.read_bytes()).hexdigest(),
          "-MonitorReceiptPath",str(monitor_path),"-FinalizeOnly"]
    actual=subprocess.run(args,capture_output=True,text=True,timeout=30,cwd=diagnostic.REPO_ROOT)
    assert actual.returncode==(0 if expected=="appearance_diagnostic_completed" else 1),actual.stderr
    summary=json.loads(Path(packet["supervisor_report"]).read_text())
    assert summary["reason"]==expected
    assert summary["status"]==("complete" if expected=="appearance_diagnostic_completed" else "incomplete")
    assert not summary["seed_authority"] and not summary["reconstruction_gain_claimed"]
    assert summary["sampled_peak_aggregate_rss_bytes"]==123456
    assert Path(packet["project_output"]).exists() and Path(packet["json_output"]).exists()
    original=Path(packet["supervisor_report"]).read_bytes()
    repeated=subprocess.run(args,capture_output=True,text=True,timeout=30,cwd=diagnostic.REPO_ROOT)
    assert repeated.returncode!=0 and Path(packet["supervisor_report"]).read_bytes()==original


def _truncate(capture,entry,*,count=65,status="solved"):
    key=next(iter(capture.pairs));observation=capture.pairs[key]
    full=np.resize(np.asarray(observation.matches,np.float64),(count,4))
    digest=diagnostic._array_sha256(full,diagnostic._MATCH_DIGEST_DOMAIN)
    reason="" if status=="solved" else "insufficient_matches"
    observation=replace(observation,source_count=count,complete=False,
        matches=tuple(tuple(float(v) for v in row) for row in full[:64]),
        matches_sha256=digest,status=status,reason=reason)
    capture.pairs[key]=observation
    source=entry.graph.evidence[0]
    source.matches_sha256=digest;source.match_count=count;source.status=status;source.reason=reason
    return key,observation


@pytest.mark.parametrize("count,status",[(65,"solved"),(96,"solved"),(4096,"solved"),(65,"refused")])
def test_cross_view_run02_truncation_is_accounted_never_modeled(monkeypatch,count,status):
    capture,entry=_capture();key,observation=_truncate(capture,entry,count=count,status=status)
    monkeypatch.setattr(diagnostic,"diagnose_plane_union",lambda *args:pytest.fail("truncated prefix reached model/helper"))
    result=diagnostic.summarize(capture,entry)
    assert result["status"]=="complete" and result["capture_census_reconciled"]
    assert result["pair_capture_counts"]=={"capture_truncated_over64":1}
    assert result["pair_skip_counts"]=={"capture_truncated_over64":1}
    assert result["pair_status_counts"]=={status:1}
    assert result["pair_rows"]==[dict(left_id=key[0],right_id=key[1],source_count=count,
        captured_count=64,status=status,source_reason=observation.reason,
        capture_status="capture_truncated_over64",model_skip_reason="capture_truncated_over64")]
    assert result["complete_solved_pairs"]==0
    assert result["unions_evaluated"]==result["correspondences_evaluated"]==result["hypotheses_evaluated"]==0
    assert diagnostic._candidates(capture)==[]
    assert not result["seed_authority"] and not result["reconstruction_gain_claimed"]


@pytest.mark.parametrize("kind,reason",[
    ("short_prefix","malformed_correspondence_capture"),
    ("claimed_complete","malformed_correspondence_capture"),
    ("wrong_row_shape","malformed_correspondence_rows"),
    ("nan","malformed_correspondence_rows"),
    ("bool_coordinate","malformed_correspondence_rows"),
    ("source_count","source_pair_evidence_mismatch"),
    ("source_hash","source_pair_evidence_mismatch"),
    ("frame_session","source_pair_evidence_mismatch"),
    ("confidence","source_pair_evidence_mismatch")])
def test_cross_view_run02_malformed_truncated_capture_fails_closed(kind,reason):
    capture,entry=_capture();key,observation=_truncate(capture,entry)
    changes={}
    if kind=="short_prefix":changes["matches"]=observation.matches[:63]
    if kind=="claimed_complete":changes["complete"]=True
    if kind=="source_count":changes["source_count"]=66
    if kind=="source_hash":changes["matches_sha256"]="f"*64
    if kind=="frame_session":changes["session_sha256"]="f"*64
    if kind=="confidence":changes["parallax_degrees"]=100.
    if kind in {"wrong_row_shape","nan","bool_coordinate"}:
        rows=list(observation.matches)
        row=list(rows[0])
        if kind=="wrong_row_shape":row.pop()
        if kind=="nan":row[0]=float("nan")
        if kind=="bool_coordinate":row[0]=True
        rows[0]=tuple(row);changes["matches"]=tuple(rows)
    capture.pairs[key]=replace(observation,**changes)
    result=diagnostic.summarize(capture,entry)
    assert result["status"]=="incomplete" and result["denials"]=={reason:1}
    assert not result["capture_census_reconciled"]
    assert result["unions_evaluated"]==result["hypotheses_evaluated"]==0
    assert result["pair_capture_counts"]=={} and result["pair_rows"]==[]


def test_cross_view_run02_complete_digest_and_bool_count_not_laundered():
    capture,entry=_capture();key=next(iter(capture.pairs));observation=capture.pairs[key]
    rows=list(observation.matches);first=list(rows[0]);first[0]+=.01;rows[0]=tuple(first)
    capture.pairs[key]=replace(observation,matches=tuple(rows))
    result=diagnostic.summarize(capture,entry)
    assert result["status"]=="incomplete" and result["denials"]=={"pair_coordinate_digest_mismatch":1}
    with pytest.raises(diagnostic.StopDiagnostic,match="malformed_correspondence_capture"):
        diagnostic._capture_state(replace(observation,source_count=True))
    with pytest.raises(diagnostic.StopDiagnostic,match="incomplete_correspondences"):
        diagnostic._capture_state(replace(observation,complete=False))


def test_cross_view_run02_salvages_complete_pair_without_partial_prefix(monkeypatch):
    capture,entry=_capture();original_key=next(iter(capture.pairs));complete=capture.pairs[original_key]
    second=replace(complete,right_id="third",right_pixel_sha256=entry.session.frames[2].image_sha256,
                   right_shape=entry.session.frames[2].image_shape)
    capture.pairs[second.left_id,second.right_id]=second
    original_source=entry.graph.evidence[0]
    second_source=SimpleNamespace(**vars(original_source));second_source.right_id="third"
    entry.graph.evidence=(original_source,second_source)
    _truncate(capture,entry,count=96)
    original=diagnostic.diagnose_plane_union;seen=[]
    def helper(observation,*args):
        assert observation.complete is True and observation.source_count<=64
        seen.append((observation.left_id,observation.right_id))
        return original(observation,*args)
    monkeypatch.setattr(diagnostic,"diagnose_plane_union",helper)
    result=diagnostic.summarize(capture,entry)
    assert result["status"]=="complete" and result["capture_census_reconciled"]
    assert result["pair_capture_counts"]=={"capture_truncated_over64":1,"complete":1}
    assert result["pair_status_counts"]=={"solved":2}
    assert result["complete_solved_pairs"]==1 and result["unions_evaluated"]==1
    assert result["status_counts"]=={"appearance_only":1}
    assert result["hypotheses_evaluated"]==36
    assert seen==[("left","third"),("left","third")]
    assert diagnostic._candidates(capture)==[(-16,"left","third",0,1)]


def test_cross_view_run02_full_fifteen_pair_census_scalar_bound():
    capture,entry=_capture();_,observation=_truncate(capture,entry)
    capture.pairs.clear();sources=[]
    frames=entry.session.frames
    for i,left in enumerate(frames):
        for right in frames[i+1:]:
            sample=replace(observation,left_id=left.frame_id,right_id=right.frame_id,
                left_pixel_sha256=left.image_sha256,right_pixel_sha256=right.image_sha256,
                left_shape=left.image_shape,right_shape=right.image_shape)
            capture.pairs[sample.left_id,sample.right_id]=sample
            source=SimpleNamespace(**vars(entry.graph.evidence[0]))
            source.left_id=sample.left_id;source.right_id=sample.right_id;sources.append(source)
    entry.graph.evidence=tuple(sources)
    result=diagnostic.summarize(capture,entry)
    assert result["status"]=="complete" and len(result["pair_rows"])==15
    assert result["pair_capture_counts"]=={"capture_truncated_over64":15}
    assert result["pair_skip_counts"]=={"capture_truncated_over64":15}
    assert result["unions_evaluated"]==result["hypotheses_evaluated"]==0
    assert len(json.dumps(result,sort_keys=True,separators=(",",":"),allow_nan=False).encode("ascii"))<16384
    assert all(set(row)=={"left_id","right_id","source_count","captured_count","status",
                         "source_reason","capture_status","model_skip_reason"} for row in result["pair_rows"])
