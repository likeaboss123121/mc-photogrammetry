"""Structural replay is distinct from private live geometry admission."""
import dataclasses

import pytest

from mc_photogrammetry.pipeline import image_derived_graph_policy_v5 as subject
from mc_photogrammetry.pipeline import image_derived_graph_policy_v4 as phase_b
from mc_photogrammetry.pipeline import image_derived_joint_graph as graph_mod
from mc_photogrammetry.pipeline import image_derived_axis_support as axis_mod
from mc_photogrammetry.pipeline import image_derived_phase_evidence as phase_mod
from mc_photogrammetry.reconstruction import plane_graph, plane_placement as placement
from test_image_derived_joint_graph import sources, _calls


def test_live_bind_has_no_additional_matcher_selector_or_evaluator(sources,monkeypatch):
    counts = _calls(monkeypatch)
    graph = graph_mod.derive_image_derived_graph(**sources)
    result = subject.bind_image_derived_graph_policy(graph_result=graph)
    assert isinstance(result, subject.ImageDerivedPolicyBoundGraph), result
    assert result.status == "bound"
    assert counts == {"matcher": 1, "selector": 1, "evaluator": 1}
    assert result.graph_result.candidate_matches is graph.candidate_matches


def test_structural_graph_copy_cannot_acquire_live_admission(sources,monkeypatch):
    graph = graph_mod.derive_image_derived_graph(**sources)
    clone = dataclasses.replace(graph)
    assert clone == graph and clone is not graph
    monkeypatch.setattr(plane_graph,"match_plane_graph",lambda *a,**k:pytest.fail("bind rematched"))
    monkeypatch.setattr(plane_graph,"_evaluate_assignment",lambda *a,**k:pytest.fail("bind reevaluated"))
    assert isinstance(subject.bind_image_derived_graph_policy(graph_result=clone),placement.PlacementRefusal)
    assert isinstance(axis_mod.derive_image_derived_axis_support(graph_result=clone),placement.PlacementRefusal)


def test_roundtrip_structural_policy_can_explicitly_replay_sources(sources,monkeypatch):
    original = subject.bind_image_derived_graph_policy(graph_result=graph_mod.derive_image_derived_graph(**sources))
    copied_graph = dataclasses.replace(original.graph_result)
    copied_policy = dataclasses.replace(original,graph_result=copied_graph)
    assert copied_policy.result_sha256 == original.result_sha256
    # Structural constructors allow ingress; execution must not trust them.
    assert isinstance(phase_mod.derive_image_derived_phase_evidence(policy_graph_result=copied_policy),placement.PlacementRefusal)
    counts = _calls(monkeypatch)
    assert subject.replay_image_derived_graph_policy(policy_graph=copied_policy,**sources) is copied_policy
    assert counts == {"matcher":1,"selector":1,"evaluator":1}
    assert subject.bind_image_derived_graph_policy(graph_result=copied_graph).status == "bound"


@pytest.mark.parametrize("field,value",[("schema","mc-photogrammetry-image-derived-graph-policy-v4"),
    ("producer","m9-j-image-derived-graph-policy-4")])
def test_phase_b_policy_identity_cannot_reseal_to_production(sources,field,value):
    graph = graph_mod.derive_image_derived_graph(**sources)
    policy = subject.bind_image_derived_graph_policy(graph_result=graph)
    with pytest.raises(ValueError):
        dataclasses.replace(policy,**{field:value},result_sha256="")


def test_live_receipt_expiry_and_capacity_fail_closed(sources,monkeypatch):
    graph = graph_mod.derive_image_derived_graph(**sources)
    clock = graph_mod.time.monotonic()
    monkeypatch.setattr(graph_mod.time,"monotonic",lambda:clock+graph_mod._LIVE_RECEIPT_SECONDS+1)
    assert isinstance(subject.bind_image_derived_graph_policy(graph_result=graph),placement.PlacementRefusal)
    monkeypatch.setattr(graph_mod,"_MAX_LIVE_RECEIPTS",0)
    refused = graph_mod.derive_image_derived_graph(**sources)
    assert isinstance(refused,placement.PlacementRefusal)
