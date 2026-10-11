"""Focused controls for the image-derived P1-B adapter."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_image_derived_graph_policy import _policy_graph  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_graph_policy as j  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_phase_evidence as subject  # noqa: E402
from mc_photogrammetry.reconstruction import plane_placement, plane_residue  # noqa: E402


@pytest.fixture(scope="module")
def records():
    graph = _policy_graph()
    bound = j.bind_image_derived_graph_policy(graph_result=graph)
    assert bound.status == "bound"
    return graph, bound


def _forge(value, **changes):
    forged = object.__new__(type(value))
    for field in dataclasses.fields(value):
        object.__setattr__(forged, field.name,
                           changes.get(field.name, getattr(value, field.name)))
    return forged


def test_complete_tuple_covers_exact_matches_and_retains_axis_refusals(records):
    graph,bound=records;result=subject.derive_image_derived_phase_evidence(policy_graph_result=bound)
    assert result.status=="complete" and result.reason==""
    assert len(result.phase_evidence)==len(graph.matches)
    indexed,_=subject.axis_adapter._reading_index(graph.readings)
    for match,evidence in zip(graph.matches,result.phase_evidence):
        assert subject._identity(evidence)==subject._match_identity(match)
        assert tuple((v.frame,v.local_axis) for v in evidence.phase_candidates)==((match.left,"u"),(match.left,"v"),(match.right,"u"),(match.right,"v"))
        subject._bind_to_match(evidence,match,indexed)
        # Missing local phase remains a native typed refusal, rather than a
        # fabricated unavailable graph or a synthesized phase measurement.
        absent={key:dataclasses.replace(value,phase_evidence=None) for key,value in indexed.items()}
        missing=plane_residue.hypothesis_phase_shift_evidence(match,absent)
        assert all(v.refusal=="phase_evidence_absent" for v in missing.phase_candidates)
    assert subject._hex(result.phase_evidence_sha256) and subject._hex(result.result_sha256)


def test_graph_v2_positive_infinity_psr_binds_without_clamping(records):
    graph, bound = records
    result = subject.derive_image_derived_phase_evidence(
        policy_graph_result=bound)
    pairs = [(match, evidence) for match, evidence in
             zip(graph.matches, result.phase_evidence)
             if match.psr == float("inf")]
    assert pairs
    indexed, _reverse = subject.axis_adapter._reading_index(graph.readings)
    for match, evidence in pairs:
        assert evidence.relative_shift.psr == float("inf")
        assert evidence.relative_shift.to_record()["psr"] == {
            "value": None, "positive_infinity": True}
        subject._bind_to_match(evidence, match, indexed)


def test_producer_called_once_per_match_with_same_reading_map(records, monkeypatch):
    graph, bound = records
    original = plane_residue.hypothesis_phase_shift_evidence
    calls = []
    def producer(match, reading_map):
        calls.append((match, reading_map))
        return original(match, reading_map)
    monkeypatch.setattr(plane_residue, "hypothesis_phase_shift_evidence", producer)
    result = subject.derive_image_derived_phase_evidence(
        policy_graph_result=bound)
    assert result.status == "complete" and len(calls) == len(graph.matches)
    assert all(call[0] is match for call, match in zip(calls, graph.matches))
    assert len({id(call[1]) for call in calls}) == 1


def test_repeat_is_deterministic(records):
    _graph, bound = records
    first = subject.derive_image_derived_phase_evidence(policy_graph_result=bound)
    second = subject.derive_image_derived_phase_evidence(policy_graph_result=bound)
    assert first == second


def test_unavailable_graph_does_not_call_producer(records,monkeypatch):
    from test_image_derived_graph import _current_unavailable
    unavailable=j.bind_image_derived_graph_policy(graph_result=_current_unavailable())
    monkeypatch.setattr(plane_residue,"hypothesis_phase_shift_evidence",lambda *_a:pytest.fail("producer ran on unavailable graph"))
    result=subject.derive_image_derived_phase_evidence(policy_graph_result=unavailable)
    assert result.status=="unavailable" and result.phase_evidence==()


def test_top_level_refusal_is_retained_as_refused_status(records, monkeypatch):
    graph, bound = records
    original = plane_residue.hypothesis_phase_shift_evidence
    refused = original(graph.matches[0], {})
    assert refused.refusal == "reading_missing"
    monkeypatch.setattr(plane_residue, "hypothesis_phase_shift_evidence",
                        lambda *_a: refused)
    # The foreign refused identity is valid for the first row but not all rows;
    # mismatch remains a provenance refusal rather than partial phase output.
    result = subject.derive_image_derived_phase_evidence(
        policy_graph_result=bound)
    assert isinstance(result, plane_placement.PlacementRefusal)


def test_honest_per_match_refusals_are_retained_atomically(records, monkeypatch):
    _graph, bound = records
    original = plane_residue.hypothesis_phase_shift_evidence
    def producer(match, reading_map):
        value = original(match, reading_map)
        return dataclasses.replace(
            value, phase_candidates=(), refusal="phase_evidence_absent")
    monkeypatch.setattr(plane_residue, "hypothesis_phase_shift_evidence", producer)
    result = subject.derive_image_derived_phase_evidence(
        policy_graph_result=bound)
    assert result.status == "refused"
    assert result.reason == "phase_evidence_refused"
    assert len(result.phase_evidence) == len(bound.graph_result.matches)
    assert all(item.refusal == "phase_evidence_absent"
               for item in result.phase_evidence)


def test_stale_j_digest_is_refused(records):
    _graph, bound = records
    result = subject.derive_image_derived_phase_evidence(
        policy_graph_result=_forge(bound, result_sha256="0" * 64))
    assert isinstance(result, plane_placement.PlacementRefusal)


def test_producer_exception_and_wrong_type_are_refused(records, monkeypatch):
    _graph, bound = records
    def fail(*_a):
        raise ValueError("failed")
    monkeypatch.setattr(plane_residue, "hypothesis_phase_shift_evidence", fail)
    assert isinstance(subject.derive_image_derived_phase_evidence(
        policy_graph_result=bound), plane_placement.PlacementRefusal)
    monkeypatch.setattr(plane_residue, "hypothesis_phase_shift_evidence",
                        lambda *_a: object())
    assert isinstance(subject.derive_image_derived_phase_evidence(
        policy_graph_result=bound), plane_placement.PlacementRefusal)


def test_forged_nested_candidate_is_rejected_on_result_replay(records):
    _graph, bound = records
    result = subject.derive_image_derived_phase_evidence(
        policy_graph_result=bound)
    hypothesis = result.phase_evidence[0]
    candidate = hypothesis.phase_candidates[0]
    forged_candidate = _forge(candidate, rational_denominator=999)
    forged_hypothesis = _forge(
        hypothesis,
        phase_candidates=(forged_candidate,) + hypothesis.phase_candidates[1:])
    with pytest.raises(ValueError):
        dataclasses.replace(
            result,
            phase_evidence=(forged_hypothesis,) + result.phase_evidence[1:],
            phase_evidence_sha256=result.phase_evidence_sha256,
            result_sha256="")


def test_reordered_collection_is_rejected(records):
    _graph, bound = records
    result = subject.derive_image_derived_phase_evidence(
        policy_graph_result=bound)
    with pytest.raises(ValueError):
        dataclasses.replace(
            result, phase_evidence=tuple(reversed(result.phase_evidence)),
            phase_evidence_sha256=subject._phase_sha(
                tuple(reversed(result.phase_evidence))), result_sha256="")


def test_public_api_has_no_k_or_manual_io_surface():
    import inspect
    assert tuple(inspect.signature(subject.derive_image_derived_phase_evidence).parameters) \
        == ("policy_graph_result",)
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("shift_domains", "manifest", "f3", "coordinate",
                      "reference", "pathlib", "open(", "subprocess", "argparse",
                      "sys.argv", "environ", "getenv", "config", "synthetic"):
        assert forbidden not in source
