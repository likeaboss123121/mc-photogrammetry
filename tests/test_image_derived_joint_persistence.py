"""Compact lineage roundtrip through real project store; no resumed geometry."""
import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest

from mc_photogrammetry.domain.models import ScanProject,BlockVolume,SCAN_PROJECT_PRODUCTION_SCHEMA_VERSION
from mc_photogrammetry.pipeline import image_derived_joint_graph as graph_mod
from mc_photogrammetry.pipeline import image_derived_production as production
from mc_photogrammetry.pipeline import image_derived_production_record as subject
from mc_photogrammetry.pipeline import image_derived_graph_policy_v5 as policy_mod
from mc_photogrammetry.pipeline import image_derived_session as session_mod
from mc_photogrammetry.project import store
from mc_photogrammetry.reconstruction import plane_placement as placement
from test_image_derived_joint_graph import sources
from test_image_derived_joint_graph import _calls


def _lineage(graph):
    return (graph.schema,graph.producer,graph.result_sha256,
            graph.selection.schema,graph.selection.producer,graph.selection.result_sha256)


def _reseal(record):
    payload={k:v for k,v in record.items() if k != "record_sha256"}
    return dict(record,record_sha256=hashlib.sha256(subject._canonical_json(payload).encode()).hexdigest())


def test_actual_result_envelope_store_roundtrip_has_lineage_not_live_admission(sources,tmp_path):
    graph=graph_mod.derive_image_derived_graph(**sources)
    assert (graph.schema,graph.producer,graph.selection.schema,graph.selection.producer) == (
        "mc-photogrammetry-image-derived-graph-v6","m9-g-image-derived-graph-10",
        "mc-photogrammetry-endpoint-selection-v4","m9-g-endpoint-selection-4")
    entry=SimpleNamespace(entry_sha256="e"*64,session=sources["session"])
    outcome=SimpleNamespace(stage="geometry",status="geometry_unavailable",reason="synthetic_downstream_refusal",detail="",
        entry_sha256=entry.entry_sha256,lattice_sha256=sources["lattice_result"].result_sha256,
        geometry_sha256="d"*64,output_sha256="",geometry=SimpleNamespace(joint_lineage=_lineage(graph)))
    envelope,volume=production._result_envelope(entry,outcome)
    project=ScanProject(name="joint-synthetic",source_paths=["a.png","b.png"],volume=volume,
        schema_version=SCAN_PROJECT_PRODUCTION_SCHEMA_VERSION,
        image_derived_session=session_mod.image_derived_session_to_record(sources["session"]),
        image_derived_production=envelope.to_record())
    path=tmp_path/'joint.mcscan'
    store.publish_project_no_clobber(project,path)
    loaded=store.load_project(path)
    replayed=subject.production_envelope_from_record(loaded.image_derived_production)
    assert replayed.to_record() == envelope.to_record()
    assert replayed.joint_lineage == _lineage(graph)
    assert loaded.volume.to_dict()["blocks"] == []
    assert "candidate_matches" not in replayed.to_record() and "selection" not in replayed.to_record()
    # A stored digest does not issue a live receipt, even for a hash-identical copy.
    clone=dataclasses.replace(graph)
    assert isinstance(policy_mod.bind_image_derived_graph_policy(graph_result=clone),placement.PlacementRefusal)


def test_actual_output_refusal_envelope_retains_prior_joint_lineage(sources):
    graph=graph_mod.derive_image_derived_graph(**sources)
    entry=SimpleNamespace(entry_sha256="e"*64,session=sources["session"])
    refusal=placement._make_refusal("invalid_input",{"field":"synthetic.output"})
    envelope=production._refusal_envelope(entry,stage="output",refusal=refusal,lattice_sha256="c"*64,
        geometry_sha256="d"*64,joint_lineage=_lineage(graph))
    assert envelope.output_sha256 == "" and envelope.joint_lineage == _lineage(graph)
    assert subject.production_envelope_from_record(envelope.to_record()).to_record() == envelope.to_record()


@pytest.mark.parametrize("field,value",[(0,"mc-photogrammetry-image-derived-graph-v2"),
    (1,"m9-g-image-derived-graph-5"),(1,"m9-g-image-derived-graph-6"),
    (3,"mc-photogrammetry-endpoint-selection-v0"),(4,"m9-g-endpoint-selection-0"),(2,"not-a-hash")])
def test_resealed_stale_or_malformed_joint_lineage_is_refused(sources,field,value):
    graph=graph_mod.derive_image_derived_graph(**sources)
    envelope=subject.build_production_envelope(stage="geometry",status="geometry_unavailable",reason="downstream",
        entry_sha256="e"*64,lattice_sha256="c"*64,geometry_sha256="d"*64,
        session_sha256=sources["session"].session_sha256,joint_lineage=_lineage(graph))
    raw=envelope.to_record();raw["joint_lineage"][field]=value
    with pytest.raises(subject.ProductionEnvelopeError):subject.production_envelope_from_record(_reseal(raw))


def test_exact_historical_envelope_replays_without_joint_or_live_authority(sources):
    current=subject.build_production_envelope(stage="entry",status="geometry_unavailable",reason="historical_entry",
        entry_sha256="e"*64,session_sha256=sources["session"].session_sha256).to_record()
    historical=dict(current,producer="ds-r9-durable-production-record-1")
    del historical["joint_lineage"]
    historical=_reseal(historical)
    replayed=subject.production_envelope_from_record(historical)
    assert replayed.to_record() == historical and replayed.joint_lineage == ()
    assert replayed.producer != subject.PRODUCTION_RECORD_PRODUCER
    assert subject.production_volume(replayed).to_dict()["blocks"] == []
    # Historical identity cannot carry the new lineage key, even empty.
    with pytest.raises(subject.ProductionEnvelopeError):
        subject.production_envelope_from_record(_reseal(dict(historical,joint_lineage=[])))


@pytest.mark.parametrize("stage",["h","j","k","malformed_h"])
def test_real_production_direct_downstream_refusal_preserves_completed_g_lineage(sources,tmp_path,monkeypatch,stage):
    from mc_photogrammetry.pipeline import image_derived_geometry_orchestration as geometry
    target=tmp_path/(stage+'.mcscan')
    entry=SimpleNamespace(entry_sha256="e"*64,session=sources["session"],graph=sources["connection_graph"],
        pose_graph=SimpleNamespace(status="ok"),source_paths=("a.png","b.png"),project_path=str(target),
        session_record=session_mod.image_derived_session_to_record(sources["session"]))
    monkeypatch.setattr(production.entry_mod,"build_image_derived_entry",lambda *a,**k:entry)
    monkeypatch.setattr(production.lattice_mod,"measure_image_derived_lattice",lambda **kw:sources["lattice_result"])
    monkeypatch.setattr(geometry.f1_mod,"derive_image_derived_signatures",lambda **kw:sources["signatures_result"])
    refusal=placement._make_refusal("invalid_input",{"field":"synthetic."+stage})
    seen=[]
    effective_stage="h" if stage == "malformed_h" else stage
    module,name={"h":(geometry.h_mod,"derive_image_derived_axis_support"),
                 "j":(geometry.j_mod,"bind_image_derived_graph_policy"),
                 "k":(geometry.k_mod,"derive_image_derived_shift_domains")}[effective_stage]
    def refused(**kwargs):
        graph=kwargs["graph_result"] if stage != "k" else kwargs["policy_graph_result"].graph_result
        assert graph.status == "assembled" and graph.producer == "m9-g-image-derived-graph-10"
        graph_mod._require_live(graph)
        if stage == "k":assert kwargs["axis_support_result"].status == "assembled"
        seen.append(graph)
        return object() if stage == "malformed_h" else refusal
    monkeypatch.setattr(module,name,refused)
    counts=_calls(monkeypatch)
    result=production.derive_image_output_from_paths([],project_path=target)
    assert len(seen)==1
    if stage == "malformed_h":
        assert isinstance(result,placement.PlacementRefusal)
        assert result.reason == "provenance_mismatch" and result.detail["field"] == "h"
    else:assert result is refusal
    assert counts == {"matcher":1,"selector":1,"evaluator":1}
    loaded=store.load_project(target)
    envelope=subject.production_envelope_from_record(loaded.image_derived_production)
    assert envelope.stage == "geometry" and envelope.status == "refused"
    assert envelope.geometry_sha256 == "" and envelope.output_sha256 == ""
    assert envelope.refusal_sha256 == result.refusal_sha256
    assert envelope.joint_lineage == _lineage(seen[0])
    assert loaded.volume.to_dict()["blocks"] == []


def test_actual_assembler_refusal_preserves_live_g_context_through_store(sources,tmp_path,monkeypatch):
    from mc_photogrammetry.pipeline import image_derived_geometry_orchestration as geometry
    graph=graph_mod.derive_image_derived_graph(**sources)
    assert graph.status == "assembled"
    records=(("f1",sources["signatures_result"]),("g",graph))
    # Isolate the final placement boundary: the genuine fixture stops at K.
    # This is an intentionally invalid bundle, never accepted later-stage evidence.
    mapping=dict.fromkeys(geometry.d_mod.REQUIRED_PLACEMENT_KEYWORDS)
    mapping["readings"]=sources["lattice_result"].readings
    mapping["split_sources"]=sources["lattice_result"].bridge.split_sources_for_assembly()
    calls=[]
    original=geometry.d_mod.assembly.assemble_placement_universe
    def assembler(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(geometry.d_mod.assembly,"assemble_placement_universe",assembler)
    captured=[]
    refusal=geometry._assemble_geometry_placement(sources["session"].session_sha256,
        sources["lattice_result"],records,tuple(mapping.items()),lambda *values:captured.append(values))
    assert isinstance(refusal,placement.PlacementRefusal) and len(calls)==1
    assert captured == [("placement",records,refusal)]
    assert geometry._joint_lineage(captured[0][1]) == _lineage(graph)
    entry=SimpleNamespace(entry_sha256="e"*64,session=sources["session"])
    envelope=production._refusal_envelope(entry,stage="geometry",refusal=refusal,
        lattice_sha256=sources["lattice_result"].result_sha256,
        joint_lineage=geometry._joint_lineage(captured[0][1]))
    project=ScanProject(name="assembler-refusal",source_paths=["a.png","b.png"],volume=subject.production_volume(envelope),
        schema_version=SCAN_PROJECT_PRODUCTION_SCHEMA_VERSION,
        image_derived_session=session_mod.image_derived_session_to_record(sources["session"]),
        image_derived_production=envelope.to_record())
    target=tmp_path/'assembler.mcscan'
    store.publish_project_no_clobber(project,target)
    loaded=subject.production_envelope_from_record(store.load_project(target).image_derived_production)
    assert loaded.joint_lineage == _lineage(graph)
    assert loaded.geometry_sha256 == loaded.output_sha256 == ""
    assert loaded.refusal_sha256 == refusal.refusal_sha256


@pytest.mark.parametrize("case",["signature","foreign_result"])
def test_final_placement_validation_refusals_preserve_completed_context(sources,monkeypatch,case):
    from mc_photogrammetry.pipeline import image_derived_geometry_orchestration as geometry
    graph=graph_mod.derive_image_derived_graph(**sources)
    records=(("f1",sources["signatures_result"]),("g",graph))
    mapping=() if case == "signature" else tuple((name,None) for name in geometry.d_mod.REQUIRED_PLACEMENT_KEYWORDS)
    if case == "foreign_result":
        monkeypatch.setattr(geometry.d_mod,"assemble_image_derived_placement",lambda **kw:object())
    captured=[]
    refusal=geometry._assemble_geometry_placement(sources["session"].session_sha256,
        sources["lattice_result"],records,mapping,lambda *values:captured.append(values))
    assert isinstance(refusal,placement.PlacementRefusal) and captured == [("placement",records,refusal)]
    assert geometry._joint_lineage(captured[0][1]) == _lineage(graph)
