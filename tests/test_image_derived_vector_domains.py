"""Focused controls for the image-derived C2-R adapter."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_image_derived_graph_policy import _policy_graph  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_axis_support as h  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_graph_policy as j  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_phase_evidence as l  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_shift_domains as k  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_vector_domain_policy as m1  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_vector_domains as subject  # noqa: E402
from mc_photogrammetry.reconstruction import (plane_placement, plane_residue,
                                               plane_shift_domain,
                                               plane_vector_domain)  # noqa: E402


@pytest.fixture(scope="module")
def records():
    graph = _policy_graph()
    axis = h.derive_image_derived_axis_support(graph_result=graph)
    bound = j.bind_image_derived_graph_policy(graph_result=graph)
    shift = k.derive_image_derived_shift_domains(
        axis_support_result=axis, policy_graph_result=bound)
    phase = l.derive_image_derived_phase_evidence(policy_graph_result=bound)
    assert shift.status == "complete" and phase.status == "complete"
    return graph, shift, phase


def _forge(value, **changes):
    forged = object.__new__(type(value))
    for field in dataclasses.fields(value):
        object.__setattr__(forged, field.name,
                           changes.get(field.name, getattr(value, field.name)))
    return forged


def test_genuine_inputs_preserve_honest_native_c2_status(records):
    _graph, shift, phase = records
    result = subject.derive_image_derived_vector_domains(
        shift_domains_result=shift, phase_evidence_result=phase)
    assert result.status == "refused"
    assert result.reason == "vector_domain_refused"
    assert result.vector_domain_result.status == "refused"
    assert result.vector_domain_result.refusals
    assert subject._hex(result.c2r_input_sha256)
    assert subject._hex(result.c2r_result_sha256)
    assert subject._hex(result.result_sha256)


def test_builder_called_once_with_exact_genuine_objects(records, monkeypatch):
    graph, shift, phase = records
    expected = subject.derive_image_derived_vector_domains(
        shift_domains_result=shift,
        phase_evidence_result=phase).vector_domain_result
    seen = {"calls": 0}
    def builder(support, p1d, matches, phases, **kwargs):
        seen.update(calls=seen["calls"] + 1, support=support, p1d=p1d,
                    matches=matches, phases=phases, kwargs=kwargs)
        return expected
    monkeypatch.setattr(plane_vector_domain, "build_lateral_vector_domains", builder)
    monkeypatch.setattr(plane_vector_domain, "validate_lateral_vector_domain_inputs",
                        lambda *_a, **_k: True)
    result = subject.derive_image_derived_vector_domains(
        shift_domains_result=shift, phase_evidence_result=phase)
    assert result.status == expected.status and seen["calls"] == 1
    assert seen["support"] is shift.axis_support_result.axis_support
    assert seen["p1d"] is shift.shift_domain_result
    assert seen["matches"] is graph.matches
    assert seen["phases"] is phase.phase_evidence
    assert seen["kwargs"]["readings"] is graph.readings
    assert seen["kwargs"]["signatures"] == tuple(
        item.signature for item in graph.signatures)
    assert seen["kwargs"]["comparison_tolerance"] == 1e-12
    assert seen["kwargs"]["budget"] is m1.IMAGE_DERIVED_VECTOR_DOMAIN_POLICY.budget


def test_repeat_is_deterministic(records):
    _graph, shift, phase = records
    first = subject.derive_image_derived_vector_domains(
        shift_domains_result=shift, phase_evidence_result=phase)
    second = subject.derive_image_derived_vector_domains(
        shift_domains_result=shift, phase_evidence_result=phase)
    assert first == second


def test_valid_shift_refusal_is_independent_and_does_not_build(
        records, monkeypatch):
    graph, shift, phase = records
    refused = plane_shift_domain.build_shift_domains(
        shift.axis_support_result.axis_support, (),
        tuple(item.signature for item in graph.signatures),
        readings=graph.readings, max_shift_cells=shift.max_shift_cells,
        budget=shift.policy_graph_result.policy.budget)
    assert refused.status == "refused"
    monkeypatch.setattr(plane_shift_domain, "build_shift_domains",
                        lambda *_a, **_k: refused)
    shifted = k.derive_image_derived_shift_domains(
        axis_support_result=shift.axis_support_result,
        policy_graph_result=shift.policy_graph_result)
    assert shifted.status == "refused"
    monkeypatch.setattr(plane_vector_domain, "build_lateral_vector_domains",
                        lambda *_a, **_k: pytest.fail("builder must not run"))
    result = subject.derive_image_derived_vector_domains(
        shift_domains_result=shifted, phase_evidence_result=phase)
    assert result.status == "unavailable"
    assert result.reason == "shift_domains_refused"
    assert result.vector_domain_result is None


def test_valid_phase_refusal_is_independent_and_does_not_build(
        records, monkeypatch):
    _graph, shift, phase = records
    original = plane_residue.hypothesis_phase_shift_evidence
    def producer(match, reading_map):
        value = original(match, reading_map)
        return dataclasses.replace(
            value, phase_candidates=(), refusal="phase_evidence_absent")
    monkeypatch.setattr(plane_residue, "hypothesis_phase_shift_evidence", producer)
    refused = l.derive_image_derived_phase_evidence(
        policy_graph_result=phase.policy_graph_result)
    assert refused.status == "refused"
    monkeypatch.setattr(plane_vector_domain, "build_lateral_vector_domains",
                        lambda *_a, **_k: pytest.fail("builder must not run"))
    result = subject.derive_image_derived_vector_domains(
        shift_domains_result=shift, phase_evidence_result=refused)
    assert result.status == "unavailable"
    assert result.reason == "phase_evidence_refused"


def test_valid_atomic_phase_unavailable_does_not_build(records, monkeypatch):
    _graph, shift, phase = records
    unavailable = dataclasses.replace(
        phase, status="unavailable", reason="upstream", phase_evidence=(),
        phase_evidence_sha256=l._phase_sha(()), result_sha256="")
    monkeypatch.setattr(plane_vector_domain, "build_lateral_vector_domains",
                        lambda *_a, **_k: pytest.fail("builder must not run"))
    result = subject.derive_image_derived_vector_domains(
        shift_domains_result=shift, phase_evidence_result=unavailable)
    assert result.status == "unavailable"
    assert result.reason == "phase_evidence_unavailable"


def test_crossed_and_stale_sources_are_refused(records):
    _graph, shift, phase = records
    crossed = _forge(phase, policy_graph_result_sha256="0" * 64)
    assert isinstance(subject.derive_image_derived_vector_domains(
        shift_domains_result=shift, phase_evidence_result=crossed),
        plane_placement.PlacementRefusal)
    stale = _forge(shift, result_sha256="0" * 64)
    assert isinstance(subject.derive_image_derived_vector_domains(
        shift_domains_result=stale, phase_evidence_result=phase),
        plane_placement.PlacementRefusal)


def test_builder_failure_wrong_type_and_validator_failure_are_refused(
        records, monkeypatch):
    _graph, shift, phase = records
    monkeypatch.setattr(plane_vector_domain, "build_lateral_vector_domains",
                        lambda *_a, **_k: object())
    assert isinstance(subject.derive_image_derived_vector_domains(
        shift_domains_result=shift, phase_evidence_result=phase),
        plane_placement.PlacementRefusal)
    expected = plane_vector_domain.build_lateral_vector_domains
    monkeypatch.setattr(plane_vector_domain, "build_lateral_vector_domains",
                        lambda *_a, **_k: expected)
    assert isinstance(subject.derive_image_derived_vector_domains(
        shift_domains_result=shift, phase_evidence_result=phase),
        plane_placement.PlacementRefusal)


def test_forged_nested_c2_is_rejected_on_result_replay(records):
    _graph, shift, phase = records
    result = subject.derive_image_derived_vector_domains(
        shift_domains_result=shift, phase_evidence_result=phase)
    forged_c2 = _forge(result.vector_domain_result, result_sha256="0" * 64)
    with pytest.raises(ValueError):
        dataclasses.replace(result, vector_domain_result=forged_c2,
                            c2r_result_sha256="0" * 64, result_sha256="")


def test_public_api_has_no_manual_or_io_surface():
    import inspect
    assert tuple(inspect.signature(
        subject.derive_image_derived_vector_domains).parameters) == (
            "shift_domains_result", "phase_evidence_result")
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("manifest", "f3", "coordinate", "reference", "pathlib",
                      "open(", "subprocess", "argparse", "sys.argv", "environ",
                      "getenv", "config", "synthetic"):
        assert forbidden not in source
