"""Focused controls for the policy-bound image-derived P1-D adapter."""
from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_image_derived_graph_policy import _policy_graph  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_axis_support as h  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_graph_policy as j  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_shift_domains as subject  # noqa: E402
from mc_photogrammetry.reconstruction import plane_graph  # noqa: E402
from mc_photogrammetry.reconstruction import plane_placement, plane_shift_domain  # noqa: E402


@pytest.fixture(scope="module")
def records():
    graph = _policy_graph()
    axis = h.derive_image_derived_axis_support(graph_result=graph)
    policy_graph = j.bind_image_derived_graph_policy(graph_result=graph)
    assert axis.status == "assembled" and policy_graph.status == "bound"
    return graph, axis, policy_graph


def _forge(value, **changes):
    forged = object.__new__(type(value))
    for field in dataclasses.fields(value):
        object.__setattr__(forged, field.name,
                           changes.get(field.name, getattr(value, field.name)))
    return forged


def test_complete_p1d_is_replayed_and_digest_bound(records):
    graph, axis, policy_graph = records
    result = subject.derive_image_derived_shift_domains(
        axis_support_result=axis, policy_graph_result=policy_graph)
    assert result.status == "complete" and result.reason == ""
    assert result.shift_domain_result.status == "complete"
    assert result.schema == subject.IMAGE_DERIVED_SHIFT_DOMAINS_SCHEMA
    assert result.producer == subject.IMAGE_DERIVED_SHIFT_DOMAINS_PRODUCER
    assert result.correlation_algorithm_id == \
        plane_graph.PLANE_CORRELATION_ALGORITHM
    assert result.max_shift_cells == 32
    assert result.shift_domain_result.schema == plane_shift_domain.SHIFT_DOMAIN_SCHEMA
    assert result.shift_domain_result.version == plane_shift_domain.SHIFT_DOMAIN_VERSION
    # The expanded FFT canvas makes this fixture's sidelobe spread measurable.
    # Assert its finite admission evidence; positive-infinity encoding has a
    # separate unchanged low-level control.
    assert all(math.isfinite(match.psr) and match.psr >= 3.0
               for match in graph.matches)
    for domain in result.shift_domain_result.domains:
        selected = next(entry for entry in domain.entries
                        if (entry.row_shift, entry.column_shift)
                        == domain.admitted_top_shift_rc)
        assert selected.overlap_cells == domain.admitted_overlap_cells
    assert result.p1d_input_sha256 == result.shift_domain_result.input_sha256
    assert subject._hex(result.p1d_result_sha256)
    assert subject._hex(result.result_sha256)
    assert plane_shift_domain.validate_shift_domain_inputs(
        result.shift_domain_result, axis.axis_support, graph.matches,
        tuple(item.signature for item in graph.signatures), readings=graph.readings,
        max_shift_cells=32)


def test_builder_called_once_with_exact_objects_and_frozen_policy(records, monkeypatch):
    graph, axis, policy_graph = records
    expected = subject.derive_image_derived_shift_domains(
        axis_support_result=axis, policy_graph_result=policy_graph).shift_domain_result
    seen = {"calls": 0}
    def builder(support, matches, signatures, **kwargs):
        seen.update(calls=seen["calls"] + 1, support=support, matches=matches,
                    signatures=signatures, kwargs=kwargs)
        return expected
    monkeypatch.setattr(plane_shift_domain, "build_shift_domains", builder)
    result = subject.derive_image_derived_shift_domains(
        axis_support_result=axis, policy_graph_result=policy_graph)
    assert result.status == "complete" and seen["calls"] == 1
    assert seen["support"] is axis.axis_support
    assert seen["matches"] is graph.matches
    assert seen["signatures"] == tuple(item.signature for item in graph.signatures)
    assert seen["kwargs"]["readings"] is graph.readings
    assert seen["kwargs"]["max_shift_cells"] == 32
    assert seen["kwargs"]["budget"] is policy_graph.policy.budget


def test_deterministic_repeat(records):
    _graph, axis, policy_graph = records
    first = subject.derive_image_derived_shift_domains(
        axis_support_result=axis, policy_graph_result=policy_graph)
    second = subject.derive_image_derived_shift_domains(
        axis_support_result=axis, policy_graph_result=policy_graph)
    assert first == second


def test_axis_unavailable_propagates_without_builder(records, monkeypatch):
    _graph, axis, policy_graph = records
    unavailable = dataclasses.replace(
        axis, status="unavailable", reason="solution_abstained",
        axis_support=None, result_sha256="")
    monkeypatch.setattr(plane_shift_domain, "build_shift_domains",
                        lambda *_a, **_k: pytest.fail("builder must not run"))
    result = subject.derive_image_derived_shift_domains(
        axis_support_result=unavailable, policy_graph_result=policy_graph)
    assert result.status == "unavailable" and result.shift_domain_result is None


def test_policy_graph_unavailable_propagates_without_builder(records, monkeypatch):
    graph, axis, policy_graph = records
    unavailable_graph = dataclasses.replace(
        graph, status="unavailable", reason="no_matches", detail="none",
        matches=(), component_solution=None, result_sha256="")
    unavailable = j.bind_image_derived_graph_policy(graph_result=unavailable_graph)
    monkeypatch.setattr(plane_shift_domain, "build_shift_domains",
                        lambda *_a, **_k: pytest.fail("builder must not run"))
    # A ready H cannot bind a different unavailable graph digest: contradiction.
    result = subject.derive_image_derived_shift_domains(
        axis_support_result=axis, policy_graph_result=unavailable)
    assert isinstance(result, plane_placement.PlacementRefusal)


def test_native_refused_and_budget_branches_remain_distinct(records, monkeypatch):
    graph, axis, policy_graph = records
    raw = tuple(item.signature for item in graph.signatures)
    refused = plane_shift_domain.build_shift_domains(
        axis.axis_support, (), raw, readings=graph.readings,
        max_shift_cells=32, budget=policy_graph.policy.budget)
    assert refused.status == "refused"
    tiny = dataclasses.replace(
        policy_graph.policy.budget, max_overlap_cell_visits=1)
    exhausted = plane_shift_domain.build_shift_domains(
        axis.axis_support, graph.matches, raw, readings=graph.readings,
        max_shift_cells=32, budget=tiny)
    assert exhausted.status == "budget_exhausted"
    monkeypatch.setattr(plane_shift_domain, "build_shift_domains",
                        lambda *_a, **_k: refused)
    wrapped = subject.derive_image_derived_shift_domains(
        axis_support_result=axis, policy_graph_result=policy_graph)
    assert wrapped.status == "refused" and wrapped.shift_domain_result is refused
    # Replay rejects a result carrying a foreign budget under the frozen policy.
    monkeypatch.setattr(plane_shift_domain, "build_shift_domains",
                        lambda *_a, **_k: exhausted)
    contradiction = subject.derive_image_derived_shift_domains(
        axis_support_result=axis, policy_graph_result=policy_graph)
    assert isinstance(contradiction, plane_placement.PlacementRefusal)


@pytest.mark.parametrize("which", ("axis", "graph"))
def test_stale_upstream_digest_is_refused(records, which):
    _source, axis, policy_graph = records
    if which == "axis":
        axis = _forge(axis, result_sha256="0" * 64)
    else:
        policy_graph = _forge(policy_graph, result_sha256="0" * 64)
    result = subject.derive_image_derived_shift_domains(
        axis_support_result=axis, policy_graph_result=policy_graph)
    assert isinstance(result, plane_placement.PlacementRefusal)


def test_cross_graph_binding_mismatch_is_refused(records):
    _graph, axis, policy_graph = records
    changed = _forge(axis, graph_sha256="f" * 64)
    result = subject.derive_image_derived_shift_domains(
        axis_support_result=changed, policy_graph_result=policy_graph)
    assert isinstance(result, plane_placement.PlacementRefusal)


@pytest.mark.parametrize("which,field,value", (
    ("axis", "schema", "mc-photogrammetry-image-derived-axis-support"),
    ("axis", "correlation_algorithm_id", "fixed-mask-v1"),
    ("axis", "max_shift_cells", 31),
    ("graph", "schema", "mc-photogrammetry-image-derived-graph-policy"),
    ("graph", "correlation_algorithm", "fixed-mask-v1"),
    ("graph", "max_shift_cells", 31),
))
def test_v1_and_mixed_upstream_provenance_is_refused(
        records, which, field, value):
    _source, axis, policy_graph = records
    if which == "axis":
        axis = _forge(axis, **{field: value})
    else:
        policy_graph = _forge(policy_graph, **{field: value})
    result = subject.derive_image_derived_shift_domains(
        axis_support_result=axis, policy_graph_result=policy_graph)
    assert isinstance(result, plane_placement.PlacementRefusal)
    assert result.reason == "provenance_mismatch"


def test_builder_exception_and_wrong_type_are_refused(records, monkeypatch):
    _graph, axis, policy_graph = records
    def fail(*_a, **_k):
        raise ValueError("failed")
    monkeypatch.setattr(plane_shift_domain, "build_shift_domains", fail)
    assert isinstance(subject.derive_image_derived_shift_domains(
        axis_support_result=axis, policy_graph_result=policy_graph),
        plane_placement.PlacementRefusal)
    monkeypatch.setattr(plane_shift_domain, "build_shift_domains",
                        lambda *_a, **_k: object())
    assert isinstance(subject.derive_image_derived_shift_domains(
        axis_support_result=axis, policy_graph_result=policy_graph),
        plane_placement.PlacementRefusal)


def test_public_api_has_no_manual_or_io_surface():
    import inspect
    assert tuple(inspect.signature(subject.derive_image_derived_shift_domains).parameters) \
        == ("axis_support_result", "policy_graph_result")
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("manifest", "f3", "coordinate", "reference", "pathlib",
                      "open(", "subprocess", "argparse", "sys.argv", "environ",
                      "getenv", "config", "synthetic"):
        assert forbidden not in source
