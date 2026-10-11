"""Focused controls for the image-derived C2-R-B closure adapter."""
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
from mc_photogrammetry.pipeline import image_derived_relative_translation_policy as n1  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_relative_translations as subject  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_shift_domains as k  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_vector_domains as m2  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_vector_domain_policy as m1  # noqa: E402
from mc_photogrammetry.reconstruction import plane_placement  # noqa: E402
from mc_photogrammetry.reconstruction import plane_relative_translation as relative  # noqa: E402
from mc_photogrammetry.reconstruction import plane_vector_domain  # noqa: E402


@pytest.fixture(scope="module")
def vector_result():
    graph = _policy_graph()
    axis = h.derive_image_derived_axis_support(graph_result=graph)
    bound = j.bind_image_derived_graph_policy(graph_result=graph)
    shift = k.derive_image_derived_shift_domains(
        axis_support_result=axis, policy_graph_result=bound)
    phase = l.derive_image_derived_phase_evidence(policy_graph_result=bound)
    result = m2.derive_image_derived_vector_domains(
        shift_domains_result=shift, phase_evidence_result=phase)
    assert result.status == "refused"
    return result


def _forge(value, **changes):
    forged = object.__new__(type(value))
    for field in dataclasses.fields(value):
        object.__setattr__(forged, field.name,
                           changes.get(field.name, getattr(value, field.name)))
    return forged


def test_genuine_image_refusal_is_atomic_and_does_not_solve(
        vector_result, monkeypatch):
    monkeypatch.setattr(relative, "solve_relative_translation_graph",
                        lambda *_a, **_k: pytest.fail("solver must not run"))
    result = subject.derive_image_derived_relative_translations(
        vector_domains_result=vector_result)
    assert result.status == "unavailable"
    assert result.reason == "vector_domains_refused"
    assert result.placement_readiness == "unavailable"
    assert result.relative_translation_result is None
    assert subject._hex(result.result_sha256)


def test_unavailable_repeat_is_deterministic(vector_result):
    first = subject.derive_image_derived_relative_translations(
        vector_domains_result=vector_result)
    second = subject.derive_image_derived_relative_translations(
        vector_domains_result=vector_result)
    assert first == second


def test_stale_and_wrong_type_inputs_are_refused(vector_result):
    stale = _forge(vector_result, result_sha256="0" * 64)
    assert isinstance(subject.derive_image_derived_relative_translations(
        vector_domains_result=stale), plane_placement.PlacementRefusal)
    assert isinstance(subject.derive_image_derived_relative_translations(
        vector_domains_result=object()), plane_placement.PlacementRefusal)


def test_deep_c2rb_replay_rejects_nested_forgery():
    from test_plane_relative_translation import _a4_triangle_case, _runtime_budget
    sources = _a4_triangle_case()
    vector, p1d, signatures, matches, support, phases, readings = sources
    result = relative.solve_relative_translation_graph(
        vector, support, p1d, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        closure_tolerance=0.0, budget=_runtime_budget())
    assert result.status == "complete"
    component = _forge(result.components[0], component_sha256="0" * 64)
    forged = _forge(result, components=(component,) + result.components[1:])
    with pytest.raises(ValueError):
        subject._replay_c2rb(forged)


def test_complete_c2r_fixture_delegates_adapter_once_with_frozen_policy(
        vector_result, monkeypatch):
    from test_plane_relative_translation import _a4_triangle_case
    _vector, p1d, signatures, matches, support, phases, readings = \
        _a4_triangle_case()
    vector = plane_vector_domain.build_lateral_vector_domains(
        support, p1d, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        budget=m1.IMAGE_DERIVED_VECTOR_DOMAIN_POLICY.budget)
    assert vector.status == "complete"
    ready = dataclasses.replace(
        vector_result, status="complete", reason="", detail="",
        vector_domain_result=vector, c2r_input_sha256=vector.input_sha256,
        c2r_result_sha256=vector.result_sha256, result_sha256="")
    expected = relative.solve_relative_translation_graph(
        vector, support, p1d, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        closure_tolerance=0.0,
        budget=n1.IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY.budget)
    assert expected.status == "complete"
    seen = {"calls": 0}
    def solver(vector_value, support_value, p1d_value, match_values,
               phase_values, **kwargs):
        seen.update(calls=seen["calls"] + 1, vector=vector_value,
                    support=support_value, p1d=p1d_value, matches=match_values,
                    phases=phase_values, kwargs=kwargs)
        return expected
    monkeypatch.setattr(relative, "solve_relative_translation_graph", solver)
    result = subject.derive_image_derived_relative_translations(
        vector_domains_result=ready)
    assert result.status == "complete"
    image_shift = ready.shift_domains_result
    image_graph = image_shift.policy_graph_result.graph_result
    assert seen["calls"] == 1
    assert seen["vector"] is vector
    assert seen["support"] is image_shift.axis_support_result.axis_support
    assert seen["p1d"] is image_shift.shift_domain_result
    assert seen["matches"] is image_graph.matches
    assert seen["phases"] is ready.phase_evidence_result.phase_evidence
    assert seen["kwargs"]["readings"] is image_graph.readings
    assert seen["kwargs"]["signatures"] == tuple(
        item.signature for item in image_graph.signatures)
    assert seen["kwargs"]["budget"] is n1.IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY.budget


def test_public_api_has_no_manual_or_io_surface():
    import inspect
    assert tuple(inspect.signature(
        subject.derive_image_derived_relative_translations).parameters) == (
            "vector_domains_result",)
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("manifest", "f3", "coordinate", "reference", "pathlib",
                      "open(", "subprocess", "argparse", "sys.argv", "environ",
                      "getenv", "config", "synthetic"):
        assert forbidden not in source
