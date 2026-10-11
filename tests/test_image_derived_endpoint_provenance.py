"""Focused controls for the image-derived V2-A adapter."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_image_derived_graph_policy import _policy_graph  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_axis_support as h  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_endpoint_provenance as subject  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_endpoint_provenance_policy as o1  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_graph_policy as j  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_phase_evidence as l  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_relative_translations as n2  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_relative_translation_policy as n1  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_shift_domains as k  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_vector_domain_policy as m1  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_vector_domains as m2  # noqa: E402
from mc_photogrammetry.reconstruction import plane_endpoint_provenance as v2a  # noqa: E402
from mc_photogrammetry.reconstruction import plane_placement  # noqa: E402
from mc_photogrammetry.reconstruction import plane_relative_translation as c2rb  # noqa: E402
from mc_photogrammetry.reconstruction import plane_vector_domain  # noqa: E402


@pytest.fixture(scope="module")
def relative_result():
    graph = _policy_graph()
    axis = h.derive_image_derived_axis_support(graph_result=graph)
    bound = j.bind_image_derived_graph_policy(graph_result=graph)
    shift = k.derive_image_derived_shift_domains(
        axis_support_result=axis, policy_graph_result=bound)
    phase = l.derive_image_derived_phase_evidence(policy_graph_result=bound)
    vector = m2.derive_image_derived_vector_domains(
        shift_domains_result=shift, phase_evidence_result=phase)
    result = n2.derive_image_derived_relative_translations(
        vector_domains_result=vector)
    assert result.status == "unavailable"
    return result


def _forge(value, **changes):
    forged = object.__new__(type(value))
    for field in dataclasses.fields(value):
        object.__setattr__(forged, field.name,
                           changes.get(field.name, getattr(value, field.name)))
    return forged


def test_genuine_chain_unavailable_is_atomic_and_does_not_analyze(
        relative_result, monkeypatch):
    monkeypatch.setattr(v2a, "analyze_endpoint_provenance",
                        lambda *_a, **_k: pytest.fail("V2-A must not run"))
    result = subject.derive_image_derived_endpoint_provenance(
        relative_translations_result=relative_result)
    assert result.status == "unavailable"
    assert result.reason == "relative_translations_unavailable"
    assert result.classification is None
    assert result.endpoint_provenance_result is None
    assert subject._hex(result.result_sha256)


def test_unavailable_repeat_is_deterministic(relative_result):
    first = subject.derive_image_derived_endpoint_provenance(
        relative_translations_result=relative_result)
    second = subject.derive_image_derived_endpoint_provenance(
        relative_translations_result=relative_result)
    assert first == second


def test_stale_and_wrong_type_inputs_are_refused(relative_result):
    stale = _forge(relative_result, result_sha256="0" * 64)
    assert isinstance(subject.derive_image_derived_endpoint_provenance(
        relative_translations_result=stale), plane_placement.PlacementRefusal)
    assert isinstance(subject.derive_image_derived_endpoint_provenance(
        relative_translations_result=object()), plane_placement.PlacementRefusal)


def test_deep_v2a_replay_rejects_nested_forgery():
    from test_plane_endpoint_provenance import ring, solved_ring, _ring_budget, _run_public
    ring_value = ring.__wrapped__()
    solved = solved_ring.__wrapped__(ring_value)
    result = _run_public(ring_value, solved, _ring_budget())
    assert result.status == "complete"
    domain = _forge(result.reading_domains[0], domain_sha256="0" * 64)
    forged = _forge(result, reading_domains=(domain,) + result.reading_domains[1:])
    with pytest.raises(ValueError):
        subject._replay_v2a(forged)


def test_ready_ring_delegates_adapter_once_with_exact_image_sources(
        relative_result, monkeypatch):
    from test_plane_endpoint_provenance import ring
    ring_value = ring.__wrapped__()
    _vector, p1d, signatures, matches, support, phases, readings = ring_value
    vector = plane_vector_domain.build_lateral_vector_domains(
        support, p1d, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        budget=m1.IMAGE_DERIVED_VECTOR_DOMAIN_POLICY.budget)
    assert vector.status == "complete"
    solved = c2rb.solve_relative_translation_graph(
        vector, support, p1d, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        closure_tolerance=0.0,
        budget=n1.IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY.budget)
    assert solved.status == "complete" and solved.placement_readiness == "ready"
    vector_wrapper = dataclasses.replace(
        relative_result.vector_domains_result, status="complete", reason="",
        detail="", vector_domain_result=vector,
        c2r_input_sha256=vector.input_sha256,
        c2r_result_sha256=vector.result_sha256, result_sha256="")
    ready = dataclasses.replace(
        relative_result, status="complete", reason="", detail="",
        placement_readiness="ready", vector_domains_result=vector_wrapper,
        vector_domains_result_sha256=vector_wrapper.result_sha256,
        relative_translation_result=solved,
        c2rb_input_sha256=solved.input_sha256,
        c2rb_result_sha256=solved.result_sha256, result_sha256="")
    expected = v2a.analyze_endpoint_provenance(
        solved, vector, support, p1d, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        closure_tolerance=0.0,
        budget=o1.IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY.budget)
    assert expected.status == "complete"
    seen = {"calls": 0}
    def analyzer(c2rb_value, vector_value, support_value, p1d_value,
                 match_values, phase_values, **kwargs):
        seen.update(calls=seen["calls"] + 1, c2rb=c2rb_value,
                    vector=vector_value, support=support_value, p1d=p1d_value,
                    matches=match_values, phases=phase_values, kwargs=kwargs)
        return expected
    monkeypatch.setattr(v2a, "analyze_endpoint_provenance", analyzer)
    result = subject.derive_image_derived_endpoint_provenance(
        relative_translations_result=ready)
    assert result.status == "complete"
    assert result.classification == expected.classification
    image_shift = vector_wrapper.shift_domains_result
    image_graph = image_shift.policy_graph_result.graph_result
    assert seen["calls"] == 1 and seen["c2rb"] is solved
    assert seen["vector"] is vector
    assert seen["support"] is image_shift.axis_support_result.axis_support
    assert seen["p1d"] is image_shift.shift_domain_result
    assert seen["matches"] is image_graph.matches
    assert seen["phases"] is vector_wrapper.phase_evidence_result.phase_evidence
    assert seen["kwargs"]["readings"] is image_graph.readings
    assert seen["kwargs"]["budget"] is o1.IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY.budget


def test_public_api_has_no_manual_or_io_surface():
    import inspect
    assert tuple(inspect.signature(
        subject.derive_image_derived_endpoint_provenance).parameters) == (
            "relative_translations_result",)
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("manifest", "f3", "coordinate", "reference", "pathlib",
                      "open(", "subprocess", "argparse", "sys.argv", "environ",
                      "getenv", "config", "synthetic"):
        assert forbidden not in source
