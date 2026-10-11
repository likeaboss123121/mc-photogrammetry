"""Closed production orchestration uses one selected graph and never legacy G."""
import dataclasses

import numpy as np
import pytest

from mc_photogrammetry.pipeline import image_derived_geometry_orchestration as subject
from mc_photogrammetry.pipeline import image_derived_graph as legacy
from mc_photogrammetry.pipeline import image_derived_joint_graph as graph_mod
from mc_photogrammetry.pipeline import image_derived_graph_policy_v5 as policy_mod
from mc_photogrammetry.pipeline import image_derived_signatures as f1
from mc_photogrammetry.reconstruction import plane_placement as placement
from test_image_derived_joint_graph import sources, _calls
from test_image_derived_graph_v3 import _source_records
from test_image_derived_connections import _simple_session, _budgets, _solved_result


def _source_with_empty_axis(sources, axis):
    original = sources["signatures_result"]
    evidence = []
    for e in original.signatures:
        if axis is None or e.reading_axis == axis:
            signature = dataclasses.replace(e.signature, coverage=f1._readonly_array(np.zeros(e.signature.coverage.shape,dtype=bool)))
            e = dataclasses.replace(e,signature=signature,signature_sha256="")
        evidence.append(e)
    return dict(sources,signatures_result=dataclasses.replace(original,signatures=tuple(evidence),
                signatures_sha256="",result_sha256=""))


@pytest.mark.parametrize("axis",[None,1])
def test_no_factors_or_unsupported_endpoint_never_reaches_geometry(sources,monkeypatch,axis):
    source = _source_with_empty_axis(sources,axis)
    counts = _calls(monkeypatch)
    graph = graph_mod.derive_image_derived_graph(**source)
    assert isinstance(graph,graph_mod.ImageDerivedGraphResult),graph
    assert graph.selection.status == "no" and graph.geometry_evaluations == 0
    assert graph.selection.proof_record()["eligibility_reasons"]
    assert counts == {"matcher":1,"selector":1,"evaluator":0}


def test_real_orchestration_binds_joint_lineage_and_stops_honestly(sources,monkeypatch):
    # Use validated synthetic source records, not a mocked graph or selector.
    monkeypatch.setattr(f1,"derive_image_derived_signatures",lambda **kw:sources["signatures_result"])
    monkeypatch.setattr(legacy,"_derive_image_derived_graph",lambda **kw:pytest.fail("legacy production G fallback"))
    counts = _calls(monkeypatch)
    result = subject.derive_image_derived_geometry(session=sources["session"],
        lattice_result=sources["lattice_result"],connection_graph=sources["connection_graph"])
    assert isinstance(result,subject.ImageDerivedGeometryResult),result
    records=dict(result.records)
    graph=records["g"]
    assert graph.producer == "m9-g-image-derived-graph-8" and graph.status == "assembled"
    assert result.joint_lineage == (graph.schema,graph.producer,graph.result_sha256,
        graph.selection.schema,graph.selection.producer,graph.selection.result_sha256)
    assert counts == {"matcher":1,"selector":1,"evaluator":1}
    # Later gates remain unchanged; this fixture need not fabricate placement.
    assert tuple(records)[:5] == ("f1","g","h","j","k")
    assert isinstance(records["j"],policy_mod.ImageDerivedPolicyBoundGraph)
    assert records["j"].graph_result is graph
    assert records["h"].status == "assembled" and records["j"].status == "bound"
    assert isinstance(records["k"],subject.k_mod.ImageDerivedShiftDomainsResult)
    assert result.status == "unavailable" and result.reason == "k_refused"


def test_multiple_orchestration_stops_at_g_with_no_downstream_calls(monkeypatch):
    sources = _source_records(monkeypatch,alternatives=True)
    monkeypatch.setattr(f1,"derive_image_derived_signatures",lambda **kw:sources["signatures_result"])
    monkeypatch.setattr(subject.h_mod,"derive_image_derived_axis_support",lambda **kw:pytest.fail("H reached on multiple"))
    counts = _calls(monkeypatch)
    result = subject.derive_image_derived_geometry(session=sources["session"],
        lattice_result=sources["lattice_result"],connection_graph=sources["connection_graph"])
    assert result.status == "unavailable" and result.reason == "g_unavailable"
    assert tuple(name for name,_ in result.records) == ("f1","g")
    assert dict(result.records)["g"].selection.status == "multiple"
    assert counts == {"matcher":1,"selector":1,"evaluator":0}


def test_each_actual_h_j_k_l_producer_is_reached_with_exact_selected_universe(sources,monkeypatch):
    counts=_calls(monkeypatch)
    graph=graph_mod.derive_image_derived_graph(**sources)
    assert graph.status == "assembled"
    called={"h":0,"k":0,"l":0}
    original_h=subject.h_mod.plane_axis_support.build_axis_support
    original_k=subject.k_mod.plane_shift_domain.build_shift_domains
    original_l=subject.l_mod.plane_residue.hypothesis_phase_shift_evidence
    raw=tuple(e.signature for e in graph.signatures)
    def h_kernel(solution,readings,matches,signatures):
        called["h"]+=1
        assert solution is graph.component_solution and readings is graph.readings and matches is graph.matches
        assert len(signatures)==len(raw) and all(a is b for a,b in zip(signatures,raw))
        return original_h(solution,readings,matches,signatures)
    def k_kernel(axis,matches,signatures,**kwargs):
        called["k"]+=1
        assert matches is graph.matches and kwargs["readings"] is graph.readings
        assert len(signatures)==len(raw) and all(a is b for a,b in zip(signatures,raw))
        return original_k(axis,matches,signatures,**kwargs)
    def l_kernel(match,indexed):
        called["l"]+=1
        assert any(match is row for row in graph.matches)
        assert indexed[(match.left,match.left_plane)] in graph.readings
        return original_l(match,indexed)
    monkeypatch.setattr(subject.h_mod.plane_axis_support,"build_axis_support",h_kernel)
    monkeypatch.setattr(subject.k_mod.plane_shift_domain,"build_shift_domains",k_kernel)
    monkeypatch.setattr(subject.l_mod.plane_residue,"hypothesis_phase_shift_evidence",l_kernel)
    h=subject.h_mod.derive_image_derived_axis_support(graph_result=graph)
    j=subject.j_mod.bind_image_derived_graph_policy(graph_result=graph)
    assert h.status == "assembled" and j.status == "bound" and j.graph_result is graph
    k=subject.k_mod.derive_image_derived_shift_domains(axis_support_result=h,policy_graph_result=j)
    assert called == {"h":1,"k":1,"l":len(graph.matches)}
    phase_calls_before_l=called["l"]
    l=subject.l_mod.derive_image_derived_phase_evidence(policy_graph_result=j)
    assert isinstance(k,subject.k_mod.ImageDerivedShiftDomainsResult),k
    assert isinstance(l,subject.l_mod.ImageDerivedPhaseEvidenceResult),l
    assert k.status == "refused" and k.reason
    assert len(l.phase_evidence)==len(graph.matches)
    assert called["l"]-phase_calls_before_l == len(graph.matches)
    assert called == {"h":1,"k":1,"l":2*len(graph.matches)}
    assert counts == {"matcher":1,"selector":1,"evaluator":1}


def test_geometry_resealing_legacy_stage_cannot_resume(sources):
    from mc_photogrammetry.pipeline import image_derived_graph_v3 as old
    graph = old.derive_image_derived_graph_v3(**sources)
    with pytest.raises(ValueError,match="stage record lacks a result digest"):
        subject._make(sources["session"].session_sha256,sources["lattice_result"].result_sha256,
          [("f1",sources["signatures_result"]),("g",graph)],status="unavailable",reason="g_unavailable")


def test_actual_four_frame_per_axis_disconnection_has_zero_geometry_calls(monkeypatch):
    from mc_photogrammetry.pipeline import image_derived_connections as connections
    from mc_photogrammetry.reconstruction import lattice_planes as lp, plane_geometry_provenance as provenance
    from test_image_derived_endpoint_selection import _evidence
    base=_source_records(monkeypatch)
    session=_simple_session(monkeypatch,("a","b","c","d"))
    connection=connections.build_image_derived_connections(session,budgets=_budgets(),
        matcher=lambda left,_li,right,_ri,**kw:_solved_result(left,right,yaw_class=0))
    assert connection.status == "ok"
    readings=[];weights=[];split_sources=[]
    original=base["lattice_result"]
    for frame in ("a","b","c","d"):
        for axis,old in enumerate(original.readings[:3]):
            reading=dataclasses.replace(old,frame=frame)
            readings.append(reading)
            weights.append(dataclasses.replace(original.bridge.structural_weights[axis],frame=frame,evidence_sha256=""))
            old_source=original.bridge.split_sources[axis]
            _,patch,budget=old_source.to_source()
            split=lp.produce_split_phase_evidence(patch,geometry=old_source.result.geometry,
                frame=frame,plane_index=axis,reading_axis=axis,source_region_id=reading.region_evidence.source_region_id,
                partition_config=lp.SPLIT_PHASE_PARTITION_CONFIG,budget=budget)
            assert split.status == "complete"
            split_sources.append(dataclasses.replace(old_source,frame=frame,result=split,source_sha256=""))
    readings=tuple(readings)
    bridge=provenance.bind_geometry_provenance(readings=readings,structural_weights=tuple(weights),split_sources=tuple(split_sources))
    assert isinstance(bridge,provenance.GeometryProvenanceBridge),bridge
    measured=dataclasses.replace(original,session_sha256=session.session_sha256,graph_sha256=connection.graph_sha256,
        frame_ids=("a","b","c","d"),readings=readings,bridge=bridge,result_sha256=None)
    evidence=[]
    for e in _evidence(readings):
        # Exact origin separation lies beyond the unchanged shift radius. Axis0
        # has a-b and c-d support; other axes keep all four views connected.
        if e.reading_axis == 0 and e.frame_id in ("c","d"):
            e=dataclasses.replace(e,signature=dataclasses.replace(e.signature,origin_uv=(100.,0.)),signature_sha256="")
        evidence.append(e)
    masks=tuple(f1.UsableMask(frame.frame_id,*frame.image_shape[:2],np.ones(frame.image_shape[:2],dtype=bool).tobytes())
        for frame in session.frames)
    signatures=dataclasses.replace(base["signatures_result"],session_sha256=session.session_sha256,
        lattice_sha256=measured.result_sha256,frame_ids=("a","b","c","d"),usable_masks=masks,
        signatures=tuple(evidence),masks_sha256="",signatures_sha256="",result_sha256="")
    counts=_calls(monkeypatch)
    result=graph_mod.derive_image_derived_graph(session=session,lattice_result=measured,
        signatures_result=signatures,connection_graph=connection)
    assert isinstance(result,graph_mod.ImageDerivedGraphResult),result
    assert result.selection.status == "no" and result.geometry_evaluations == 0
    reasons=result.selection.proof_record()["eligibility_reasons"]
    assert any("disconnected" in reason for reason in reasons),reasons
    assert {r.left for r in result.candidate_matches}|{r.right for r in result.candidate_matches} == {"a","b","c","d"}
    assert counts == {"matcher":1,"selector":1,"evaluator":0}
