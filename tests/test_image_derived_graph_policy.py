"""Focused controls for graph replay under the frozen P1-D policy."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_image_derived_axis_support import _successful_graph  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_graph_policy as subject  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_graph as graph_mod  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_axis_support as axis_support_mod  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_signatures as signatures_mod  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_shift_domain_policy as policy_mod  # noqa: E402
from mc_photogrammetry.reconstruction import plane_graph, plane_placement  # noqa: E402


def _policy_graph():
    return _successful_graph()


def _forge(value, **changes):
    forged = object.__new__(type(value))
    for field in dataclasses.fields(value):
        object.__setattr__(forged, field.name,
                           changes.get(field.name, getattr(value, field.name)))
    return forged


def test_genuine_radius_32_replay_binds_graph_and_policy():
    graph=_policy_graph();result=subject.bind_image_derived_graph_policy(graph_result=graph)
    assert type(graph) is graph_mod.ImageDerivedGraphResult
    assert graph.schema == graph_mod.IMAGE_DERIVED_GRAPH_SCHEMA == "mc-photogrammetry-image-derived-graph-v3"
    assert graph.producer == graph_mod.IMAGE_DERIVED_GRAPH_PRODUCER == "m9-g-image-derived-graph-7"
    assert type(result) is subject.ImageDerivedPolicyBoundGraph and result.status=="bound" and not result.reason
    assert result.schema==subject.IMAGE_DERIVED_GRAPH_POLICY_SCHEMA=="mc-photogrammetry-image-derived-graph-policy-v5"
    assert result.producer==subject.IMAGE_DERIVED_GRAPH_POLICY_PRODUCER=="m9-j-image-derived-graph-policy-5"
    assert result.graph_result is graph and result.graph_sha256==graph.result_sha256
    assert result.policy is policy_mod.IMAGE_DERIVED_SHIFT_DOMAIN_POLICY and result.policy_sha256==result.policy.policy_sha256
    assert result.max_shift_cells==32 and result.correlation_algorithm==plane_graph.PLANE_CORRELATION_ALGORITHM
    assert subject._hex(result.result_sha256)


def test_historical_producers_and_mixed_algorithm_graphs_fail_closed():
    graph = _policy_graph()
    for changes in (
            {"schema": "mc-photogrammetry-image-derived-graph"},
            {"producer": "m9-g-image-derived-graph-1"},
            {"producer": "m9-g-image-derived-graph-2"},
            {"producer": "m9-g-image-derived-graph-3"},
            {"producer": "m9-g-image-derived-graph-4"},
            {"correlation_algorithm": "fixed-mask-v1"},
            {"correlation_algorithm": "separate-mask-halo-v2"},
            {"max_shift_cells": 31}):
        refused = subject.bind_image_derived_graph_policy(
            graph_result=_forge(graph, **changes))
        assert isinstance(refused, plane_placement.PlacementRefusal)
        assert refused.reason == "provenance_mismatch"


def test_tampered_builtin_policy_algorithm_fails_closed(monkeypatch):
    graph = _policy_graph()
    policy = policy_mod.IMAGE_DERIVED_SHIFT_DOMAIN_POLICY
    forged = _forge(policy, correlation_algorithm_id="fixed-mask-v1")
    monkeypatch.setattr(policy_mod, "IMAGE_DERIVED_SHIFT_DOMAIN_POLICY", forged)
    refused = subject.bind_image_derived_graph_policy(graph_result=graph)
    assert isinstance(refused, plane_placement.PlacementRefusal)
    assert refused.reason == "provenance_mismatch"


@pytest.mark.parametrize("field,value", (
    ("correlation_algorithm", "fixed-mask-v1"),
    ("max_shift_cells", 31),
))
def test_adapter_refuses_tampered_graph_binding(field, value):
    graph = _policy_graph()
    refused = subject.bind_image_derived_graph_policy(
        graph_result=_forge(graph, **{field: value}))
    assert isinstance(refused, plane_placement.PlacementRefusal)
    assert refused.reason == "provenance_mismatch"


def test_graph_v2_digest_canonically_accepts_positive_infinity_psr():
    from test_image_derived_graph import _current_sources
    graph=graph_mod.derive_historical_image_derived_graph(**_current_sources())
    assert type(graph) is graph_mod.HistoricalImageDerivedGraphResult and graph.status=="assembled"
    matches=(dataclasses.replace(graph.matches[0],psr=float("inf")),*graph.matches[1:])
    first=dataclasses.replace(graph,matches=matches,result_sha256="")
    second=dataclasses.replace(graph,matches=matches,result_sha256="")
    assert first.result_sha256==second.result_sha256
    assert graph_mod._payload(first)["matches"][0][7]=={"value":None,"positive_infinity":True}
    assert isinstance(subject.bind_image_derived_graph_policy(graph_result=first),plane_placement.PlacementRefusal)


def test_matcher_is_called_once_with_exact_objects_and_only_radius(monkeypatch):
    graph=_policy_graph();original=plane_graph.match_plane_graph;seen=[]
    def matcher(readings,signatures,**kwargs):
        seen.append((readings,signatures,kwargs));return original(readings,signatures,**kwargs)
    monkeypatch.setattr(plane_graph,"match_plane_graph",matcher)
    bound=subject.bind_image_derived_graph_policy(graph_result=graph)
    assert bound.status=="bound" and seen==[]
    from mc_photogrammetry.pipeline import image_derived_graph_policy_v5 as current
    from test_image_derived_graph import _current_sources
    source=_current_sources(monkeypatch)
    assert current.replay_image_derived_graph_policy(policy_graph=bound,**source) is bound
    assert len(seen)==1
    readings,signatures,kwargs=seen[0]
    assert readings is source["lattice_result"].readings
    assert all(a is b.signature for a,b in zip(signatures,source["signatures_result"].signatures))
    assert kwargs=={"max_shift_cells":32}


def test_repeated_binding_is_deterministic():
    graph = _policy_graph()
    first = subject.bind_image_derived_graph_policy(graph_result=graph)
    second = subject.bind_image_derived_graph_policy(graph_result=graph)
    assert first == second and first.result_sha256 == second.result_sha256


def test_self_consistent_stale_graph_cannot_replay_policy_or_reach_placement(monkeypatch):
    from mc_photogrammetry.pipeline import image_derived_joint_graph as current
    from mc_photogrammetry.pipeline import image_derived_graph_policy_v5 as policy
    graph=_policy_graph();bound=subject.bind_image_derived_graph_policy(graph_result=graph)
    assert bound.status=="bound"
    assert bound.graph_sha256==graph_mod._sha(current._payload(graph))
    assert bound.result_sha256==subject._sha(policy._payload(bound))
    stale=_forge(graph,producer="m9-g-image-derived-graph-4")
    stale=_forge(stale,result_sha256=graph_mod._sha(current._payload(stale)))
    monkeypatch.setattr(plane_graph,"match_plane_graph",lambda *_a,**_k:pytest.fail("stale graph entered matcher"))
    assert isinstance(subject.bind_image_derived_graph_policy(graph_result=stale),plane_placement.PlacementRefusal)
    nested=_forge(bound,graph_result=stale,graph_sha256=stale.result_sha256)
    nested=_forge(nested,result_sha256=subject._sha(policy._payload(nested)))
    with pytest.raises(ValueError):dataclasses.replace(nested)
    assert isinstance(axis_support_mod.derive_image_derived_axis_support(graph_result=stale),plane_placement.PlacementRefusal)


def test_graph_digest_change_propagates_to_policy_with_unchanged_policy_identity():
    from test_image_derived_graph import _current_sources
    first=subject.bind_image_derived_graph_policy(graph_result=graph_mod.derive_image_derived_graph(**_current_sources(energy=True)))
    second=subject.bind_image_derived_graph_policy(graph_result=graph_mod.derive_image_derived_graph(**_current_sources(energy=False)))
    assert first.status==second.status=="bound"
    assert first.graph_sha256!=second.graph_sha256 and first.result_sha256!=second.result_sha256
    assert first.policy_sha256==second.policy_sha256
    assert first.schema==second.schema==subject.IMAGE_DERIVED_GRAPH_POLICY_SCHEMA
    assert first.producer==second.producer==subject.IMAGE_DERIVED_GRAPH_POLICY_PRODUCER


def test_unavailable_graph_does_not_call_matcher(monkeypatch):
    from test_image_derived_graph import _current_unavailable
    graph=_current_unavailable()
    monkeypatch.setattr(plane_graph,"match_plane_graph",lambda *_a,**_k:pytest.fail("bind re-matched unavailable G"))
    result=subject.bind_image_derived_graph_policy(graph_result=graph)
    assert result.status=="unavailable" and result.reason=="graph_unavailable"


def test_stale_graph_digest_is_refused():
    result = subject.bind_image_derived_graph_policy(
        graph_result=_forge(_policy_graph(), result_sha256="0" * 64))
    assert isinstance(result, plane_placement.PlacementRefusal)
    assert result.reason == "provenance_mismatch"


def test_reordered_evidence_is_refused():
    graph=_policy_graph()
    with pytest.raises(ValueError):dataclasses.replace(graph,signatures=tuple(reversed(graph.signatures)),result_sha256="")
    forged=_forge(graph,signatures=tuple(reversed(graph.signatures)))
    assert isinstance(subject.bind_image_derived_graph_policy(graph_result=forged),plane_placement.PlacementRefusal)


def test_foreign_signature_reading_is_refused():
    graph = _policy_graph()
    foreign = _forge(graph.signatures[0], frame_id="foreign")
    changed = _forge(graph, signatures=(foreign,) + graph.signatures[1:])
    assert isinstance(subject.bind_image_derived_graph_policy(graph_result=changed),
                      plane_placement.PlacementRefusal)


@pytest.mark.parametrize("change", ("drop", "reverse", "diagnostic"))
def test_any_match_tuple_difference_is_refused(change,monkeypatch):
    from mc_photogrammetry.pipeline import image_derived_graph_policy_v5 as current
    from test_image_derived_graph import _current_sources
    graph=_policy_graph();bound=subject.bind_image_derived_graph_policy(graph_result=graph)
    original=plane_graph.match_plane_graph;source=_current_sources(monkeypatch);calls=[]
    def matcher(*args,**kwargs):
        calls.append(1);rows=original(*args,**kwargs)
        if change=="drop":return rows[:-1]
        if change=="reverse":return tuple(reversed(rows))
        return (dataclasses.replace(rows[0],score=.123),)+rows[1:]
    monkeypatch.setattr(plane_graph,"match_plane_graph",matcher)
    result=current.replay_image_derived_graph_policy(policy_graph=bound,**source)
    assert isinstance(result,plane_placement.PlacementRefusal) and result.reason=="provenance_mismatch"
    assert len(calls)==1


def test_matcher_exception_and_non_tuple_are_refused(monkeypatch):
    from mc_photogrammetry.pipeline import image_derived_graph_policy_v5 as current
    from test_image_derived_graph import _current_sources
    graph=_policy_graph();bound=subject.bind_image_derived_graph_policy(graph_result=graph);source=_current_sources(monkeypatch)
    def fail(*_a,**_k):raise ValueError("explicit replay matcher failed")
    monkeypatch.setattr(plane_graph,"match_plane_graph",fail)
    assert isinstance(current.replay_image_derived_graph_policy(policy_graph=bound,**source),plane_placement.PlacementRefusal)
    monkeypatch.setattr(plane_graph,"match_plane_graph",lambda *_a,**_k:list(graph.candidate_matches))
    assert isinstance(current.replay_image_derived_graph_policy(policy_graph=bound,**source),plane_placement.PlacementRefusal)


def test_forged_builtin_policy_is_refused(monkeypatch):
    policy = policy_mod.IMAGE_DERIVED_SHIFT_DOMAIN_POLICY
    monkeypatch.setattr(policy_mod, "IMAGE_DERIVED_SHIFT_DOMAIN_POLICY",
                        _forge(policy, policy_sha256="0" * 64))
    result = subject.bind_image_derived_graph_policy(graph_result=_policy_graph())
    assert isinstance(result, plane_placement.PlacementRefusal)


def test_public_api_has_no_external_policy_or_io_surface():
    import inspect
    assert tuple(inspect.signature(subject.bind_image_derived_graph_policy).parameters) \
        == ("graph_result",)
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("manifest", "f3", "coordinate", "reference", "pathlib",
                      "open(", "subprocess", "argparse", "sys.argv", "environ",
                      "getenv", "config", "synthetic"):
        assert forbidden not in source


@pytest.mark.parametrize("changes", (
    {"schema": "mc-photogrammetry-image-derived-graph-policy-v2"},
    {"producer": "m9-j-image-derived-graph-policy-2"},
    {"correlation_algorithm": "separate-mask-halo-v2"},
))
def test_old_policy_bound_graph_identity_is_rejected(changes):
    bound = subject.bind_image_derived_graph_policy(graph_result=_policy_graph())
    assert bound.status == "bound"
    with pytest.raises(ValueError, match="malformed"):
        dataclasses.replace(bound, **changes, result_sha256="")
