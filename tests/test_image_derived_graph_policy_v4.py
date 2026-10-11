"""Phase B policy replay stays unavailable and rejects resealed old wrappers."""
import dataclasses

import pytest

from mc_photogrammetry.pipeline import image_derived_graph_policy_v4 as subject
from mc_photogrammetry.pipeline import image_derived_graph_policy as old_policy
from mc_photogrammetry.pipeline import image_derived_graph_v3 as graph_mod
from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
from mc_photogrammetry.pipeline import image_derived_shift_domain_policy as policy_mod
from mc_photogrammetry.pipeline import image_derived_axis_support as axis_support
from mc_photogrammetry.reconstruction import plane_graph
from test_image_derived_endpoint_selection import _forge
from test_image_derived_graph_v3 import sources, _graph, _reseal_graph, _refused


def _bound(sources):
    graph = _graph(sources)
    result = subject.bind_image_derived_graph_policy_v4(graph_result=graph, **sources)
    assert isinstance(result, subject.ImageDerivedPolicyGraphV4), result
    return result


def test_source_replayed_policy_wrapper_is_never_bound_or_geometry(sources, monkeypatch):
    monkeypatch.setattr(axis_support,"_bind_universe",lambda *a, **k: pytest.fail("forbidden axis binder"))
    monkeypatch.setattr(plane_graph,"solve_component",lambda *a, **k: pytest.fail("forbidden solver"))
    monkeypatch.setattr(plane_graph,"_evaluate_assignment",lambda *a, **k: pytest.fail("forbidden evaluator"))
    result = _bound(sources)
    assert result.schema == "mc-photogrammetry-image-derived-graph-policy-v4"
    assert result.producer == "m9-j-image-derived-graph-policy-4"
    assert result.status == "unavailable" and result.reason == "joint_geometry_not_evaluated"
    assert result.graph_result.matches == () and result.graph_result.component_solution is None
    assert result.graph_result.selection.status == "unique"
    assert result.policy is policy_mod.IMAGE_DERIVED_SHIFT_DOMAIN_POLICY
    assert subject.replay_image_derived_graph_policy_v4(policy_graph=result, **sources) is result


@pytest.mark.parametrize("field,value", [("schema", "mc-photogrammetry-image-derived-graph-policy-v3"),
                                         ("producer", "m9-j-image-derived-graph-policy-3"), ("status", "bound")])
def test_resealed_old_policy_identity_or_bound_state_cannot_promote(sources, field, value):
    result = _bound(sources)
    stale = _forge(result, **{field:value})
    stale = _forge(stale,result_sha256=selection._sha(subject._payload(stale)))
    with pytest.raises(ValueError): dataclasses.replace(stale)
    _refused(subject.replay_image_derived_graph_policy_v4(policy_graph=stale, **sources))


def test_old_wrapper_object_is_refused_without_relabeling_history(sources):
    result = _bound(sources)
    historical = object.__new__(old_policy.ImageDerivedPolicyBoundGraph)
    _refused(subject.replay_image_derived_graph_policy_v4(policy_graph=historical, **sources))
    assert old_policy.IMAGE_DERIVED_GRAPH_POLICY_SCHEMA == "mc-photogrammetry-image-derived-graph-policy-v3"
    assert old_policy.IMAGE_DERIVED_GRAPH_POLICY_PRODUCER == "m9-j-image-derived-graph-policy-3"


def test_stale_nested_graph_resealed_outer_policy_is_refused_before_matcher(sources, monkeypatch):
    result = _bound(sources)
    graph = _forge(result.graph_result,producer="m9-g-image-derived-graph-5")
    graph = _forge(graph,result_sha256=selection._sha(graph_mod._payload(graph)))
    stale = _forge(result,graph_result=graph,graph_sha256=graph.result_sha256)
    stale = _forge(stale,result_sha256=selection._sha(subject._payload(stale)))
    monkeypatch.setattr(plane_graph,"match_plane_graph",lambda *a, **k: pytest.fail("stale nested graph"))
    _refused(subject.replay_image_derived_graph_policy_v4(policy_graph=stale, **sources))


def test_valid_looking_policy_graph_hash_does_not_skip_candidate_source_replay(sources):
    result = _bound(sources)
    graph = result.graph_result
    rows = tuple(reversed(graph.candidate_matches))
    cert = selection.select_endpoint_hypotheses(readings=graph.readings,signatures=graph.signatures,
        candidate_matches=rows,frame_ids=graph.frame_ids,quadrant_constraints=graph.selection.quadrant_constraints,
        budget=graph.selection.budget,session_sha256=graph.session_sha256,lattice_sha256=graph.lattice_sha256,
        signatures_sha256=graph.signatures_sha256,connection_sha256=graph.connection_sha256)
    forged_graph = _reseal_graph(graph,candidate_matches=rows,selection=cert,reason=cert.reason,detail=cert.status)
    wrapper = dataclasses.replace(result,graph_result=forged_graph,graph_sha256=forged_graph.result_sha256,result_sha256="")
    assert wrapper.result_sha256 != result.result_sha256
    _refused(subject.replay_image_derived_graph_policy_v4(policy_graph=wrapper, **sources))


def test_policy_replay_matcher_receives_only_fixed_radius_and_no_solver_parameters(sources,monkeypatch):
    graph = _graph(sources)
    actual = plane_graph.match_plane_graph
    calls=[]
    def matched(readings,signatures,**kwargs):
        calls.append(kwargs)
        return actual(readings,signatures,**kwargs)
    monkeypatch.setattr(plane_graph,"match_plane_graph",matched)
    result=subject.bind_image_derived_graph_policy_v4(graph_result=graph,**sources)
    assert isinstance(result,subject.ImageDerivedPolicyGraphV4)
    assert calls == [{"max_shift_cells":32}]


def test_malformed_builtin_policy_refuses_before_source_matching(sources,monkeypatch):
    graph = _graph(sources)
    bad=_forge(policy_mod.IMAGE_DERIVED_SHIFT_DOMAIN_POLICY,max_shift_cells=31)
    monkeypatch.setattr(policy_mod,"IMAGE_DERIVED_SHIFT_DOMAIN_POLICY",bad)
    monkeypatch.setattr(plane_graph,"match_plane_graph",lambda *a,**k:pytest.fail("invalid policy"))
    _refused(subject.bind_image_derived_graph_policy_v4(graph_result=graph,**sources))
