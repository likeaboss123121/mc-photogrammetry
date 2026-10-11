"""Focused controls for the image-derived C1 adapter."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_image_derived_lattice_measure import _result  # noqa: E402
from test_plane_axis_support import _fixture as c1_fixture  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_axis_support as subject  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_graph as graph_mod  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_signatures as f1  # noqa: E402
from mc_photogrammetry.reconstruction import plane_axis_support, plane_graph  # noqa: E402
from mc_photogrammetry.reconstruction import plane_placement  # noqa: E402


@pytest.fixture(scope="module")
def graph_record():
    from test_image_derived_graph import _current_graph
    graph=_current_graph()
    assert graph.status == "assembled" and graph.producer == "m9-g-image-derived-graph-7"
    return graph


def _forge(value, **changes):
    forged = object.__new__(type(value))
    for field in dataclasses.fields(value):
        object.__setattr__(forged, field.name,
                           changes.get(field.name, getattr(value, field.name)))
    return forged


def _support(frame="f0"):
    return plane_axis_support.C1AxisSupport(
        refusal="solution_abstained", frames=(), global_anchor=None,
        normal_face_constraints=(), mapped_cell_shifts=(), match_refusals=(),
        axis_components=(), frame_support=())


def _successful_graph():
    from test_image_derived_graph import _current_graph
    graph=_current_graph()
    assert graph.status == "assembled"
    return graph


def test_exact_graph_objects_reach_c1_once(graph_record, monkeypatch):
    seen = {"calls": 0}
    def builder(solution, readings, matches, signatures):
        seen.update(calls=seen["calls"] + 1, solution=solution,
                    readings=readings, matches=matches, signatures=signatures)
        return _support()
    monkeypatch.setattr(plane_axis_support, "build_axis_support", builder)
    result = subject.derive_image_derived_axis_support(graph_result=graph_record)
    assert result.status == "unavailable" and result.reason == "solution_abstained"
    assert seen["calls"] == 1 and seen["solution"] is graph_record.component_solution
    assert seen["readings"] is graph_record.readings
    assert seen["matches"] is graph_record.matches
    assert seen["signatures"] == tuple(
        item.signature for item in graph_record.signatures)


def test_successful_c1_is_retained_and_digest_bound():
    graph = _successful_graph()
    result = subject.derive_image_derived_axis_support(graph_result=graph)
    assert isinstance(result, subject.ImageDerivedAxisSupportResult)
    assert result.status == "assembled" and result.reason == ""
    assert result.axis_support.refusal is None
    assert result.axis_support.frames == graph.component_solution.frames
    assert result.graph_sha256 == graph.result_sha256
    assert result.schema == subject.IMAGE_DERIVED_AXIS_SUPPORT_SCHEMA
    assert result.producer == subject.IMAGE_DERIVED_AXIS_SUPPORT_PRODUCER
    assert result.correlation_algorithm_id == \
        plane_graph.PLANE_CORRELATION_ALGORITHM
    assert result.max_shift_cells == graph.max_shift_cells
    assert subject._hex(result.result_sha256)


def test_positive_infinity_psr_is_bound_without_mutating_graph():
    graph=_successful_graph();before=graph.result_sha256
    matches=(dataclasses.replace(graph.matches[0],psr=float("inf")),*graph.matches[1:])
    support=plane_axis_support.build_axis_support(graph.component_solution,graph.readings,matches,
                                                 tuple(e.signature for e in graph.signatures))
    assert support.refusal is None
    supports=[mapped.support for mapped in support.mapped_cell_shifts if mapped.match_identity==plane_axis_support.C1MatchIdentity.from_match(matches[0])]
    assert supports and all(v.psr==float("inf") for v in supports)
    assert all(v.to_record()["psr"]=={"value":None,"positive_infinity":True} for v in supports)
    assert graph.result_sha256==before
    assert isinstance(subject.derive_image_derived_axis_support(graph_result=dataclasses.replace(graph)),plane_placement.PlacementRefusal)


@pytest.mark.parametrize("field,value", (
    ("schema", "mc-photogrammetry-image-derived-graph"),
    ("producer", "m9-g-image-derived-graph-1"),
    ("correlation_algorithm", "fixed-mask-v1"),
    ("max_shift_cells", 31),
))
def test_v1_and_mixed_graph_provenance_is_typed_refusal(field, value):
    result = subject.derive_image_derived_axis_support(
        graph_result=_forge(_successful_graph(), **{field: value}))
    assert isinstance(result, plane_placement.PlacementRefusal)
    assert result.reason == "provenance_mismatch"


def test_upstream_unavailable_is_propagated_without_calling_c1(graph_record,monkeypatch):
    from test_image_derived_graph import _current_unavailable
    unavailable=_current_unavailable()
    monkeypatch.setattr(plane_axis_support,"build_axis_support",lambda *_a:pytest.fail("C1 ran after selection abstention"))
    result=subject.derive_image_derived_axis_support(graph_result=unavailable)
    assert result.status=="unavailable" and result.reason=="graph_unavailable" and result.axis_support is None


def test_stale_graph_digest_is_a_typed_refusal(graph_record):
    stale = _forge(graph_record, result_sha256="0" * 64)
    result = subject.derive_image_derived_axis_support(graph_result=stale)
    assert isinstance(result, plane_placement.PlacementRefusal)
    assert result.reason == "provenance_mismatch"


def test_resealed_reordered_evidence_is_refused(graph_record):
    with pytest.raises(ValueError):dataclasses.replace(graph_record,signatures=tuple(reversed(graph_record.signatures)),result_sha256="")
    graph=_forge(graph_record,signatures=tuple(reversed(graph_record.signatures)))
    assert isinstance(subject.derive_image_derived_axis_support(graph_result=graph),plane_placement.PlacementRefusal)


def test_foreign_evidence_reading_is_refused(graph_record):
    evidence = graph_record.signatures[0]
    foreign = _forge(evidence, frame_id="foreign")
    graph = _forge(graph_record, signatures=(foreign,) + graph_record.signatures[1:])
    result = subject.derive_image_derived_axis_support(graph_result=graph)
    assert isinstance(result, plane_placement.PlacementRefusal)


def test_foreign_match_endpoint_is_refused(graph_record):
    match=dataclasses.replace(graph_record.matches[0],left_plane=999)
    with pytest.raises(ValueError):dataclasses.replace(graph_record,matches=(match,),result_sha256="")
    assert isinstance(subject.derive_image_derived_axis_support(graph_result=_forge(graph_record,matches=(match,))),plane_placement.PlacementRefusal)


def test_component_solution_binding_is_checked(graph_record):
    solution=dataclasses.replace(graph_record.component_solution,frames=("foreign",))
    with pytest.raises(ValueError):dataclasses.replace(graph_record,component_solution=solution,result_sha256="")
    assert isinstance(subject.derive_image_derived_axis_support(graph_result=_forge(graph_record,component_solution=solution)),plane_placement.PlacementRefusal)


def test_public_api_has_no_manual_or_io_surface():
    import inspect
    assert tuple(inspect.signature(subject.derive_image_derived_axis_support).parameters) \
        == ("graph_result",)
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("manifest", "f3", "coordinate", "reference", "pathlib",
                      "open(", "subprocess", "argparse", "sys.argv", "synthetic"):
        assert forbidden not in source
