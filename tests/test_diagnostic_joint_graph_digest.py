"""Source-bound graph digest controls for the image geometry diagnostic."""

import pytest

from scripts.evaluation import diagnose_image_geometry_failure as diagnostic
from mc_photogrammetry.pipeline import image_derived_graph as legacy_graph
from mc_photogrammetry.pipeline import image_derived_joint_graph as joint_graph
from mc_photogrammetry.reconstruction import plane_placement
from test_image_derived_graph_v3 import _source_records


def test_diagnostic_hashes_the_exact_joint_producer_payload():
    monkeypatch = pytest.MonkeyPatch()
    try:
        sources = _source_records(monkeypatch)
        graph = joint_graph.derive_image_derived_graph(**sources)
        assert type(graph) is joint_graph.ImageDerivedGraphResult

        payload = diagnostic._verified_joint_graph_payload(graph)

        assert diagnostic._digest(payload) == graph.result_sha256
        assert diagnostic._digest(legacy_graph._payload(graph)) != graph.result_sha256
    finally:
        monkeypatch.undo()


def test_diagnostic_rejects_a_tampered_joint_graph_digest():
    monkeypatch = pytest.MonkeyPatch()
    try:
        graph = joint_graph.derive_image_derived_graph(**_source_records(monkeypatch))
        object.__setattr__(graph, "result_sha256", "0" * 64)
        with pytest.raises(diagnostic.DiagnosticError, match="graph_digest") as error:
            diagnostic._verified_joint_graph_payload(graph)
        assert error.value.reason == "binding_mismatch"
    finally:
        monkeypatch.undo()


def test_diagnostic_rejects_non_joint_graph_types():
    refusal = plane_placement._make_refusal("invalid_input", {"field": "graph"})
    with pytest.raises(diagnostic.DiagnosticError, match="graph_type") as error:
        diagnostic._verified_joint_graph_payload(refusal)
    assert error.value.reason == "invalid_evidence"
