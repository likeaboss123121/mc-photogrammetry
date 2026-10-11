from types import SimpleNamespace

import pytest

from mc_photogrammetry import evaluation_surface_support as support_mod
from mc_photogrammetry import synthetic_targeted_capture_preflight as subject
from test_evaluation_surface_support import _payload_fixture
from test_synthetic_targeted_capture_preflight import _evidence
import numpy as np


def test_exact_payload_crosswalk_publishes_deterministic_surface_report():
    payload, left, right, left_signature, right_signature = _payload_fixture()
    first = subject._evaluate_surface_support_payload(
        payload, (left, right), (left_signature, right_signature), 9)
    second = subject._evaluate_surface_support_payload(
        payload, (left, right), (left_signature, right_signature), 9)
    assert first == second and len(first) == 1
    assert isinstance(first[0], support_mod.SurfaceLocalSupportReportV1)
    assert first[0].same_surface_pairs == 9


def test_crosswalk_missing_signature_and_cap_fail_atomically():
    payload, left, right, left_signature, right_signature = _payload_fixture()
    missing = subject._evaluate_surface_support_payload(
        payload, (left, right), (left_signature,), 9)
    assert missing.status == "refused" and missing.stage == "surface_support"
    capped = subject._evaluate_surface_support_payload(
        payload, (left, right), (left_signature, right_signature), 8)
    assert isinstance(capped, support_mod.SurfaceLocalSupportRefusal)
    assert capped.reason == "budget_exhausted"


def test_preflight_surface_sink_is_safe_and_missing_payload_is_typed(
        monkeypatch, tmp_path):
    evidences = tuple(_evidence(np.ones((4, 4), dtype=bool)) for _ in range(4))
    monkeypatch.setattr(subject.entry_mod, "build_image_derived_entry",
                        lambda *_a, **_k: SimpleNamespace(
                            graph=SimpleNamespace(status="unavailable"),
                            pose_graph=SimpleNamespace(status="unavailable"),
                            reason="no_connections", entry_sha256="4" * 64))
    sink = []
    report = subject.preflight_synthetic_targeted_capture(
        evidences=evidences, frame_ids=("a", "b", "c", "context"),
        project_path=tmp_path / "project.json", surface_support_sink=sink)
    assert report.status == "insufficient_evidence"
    assert len(sink) == 1 and sink[0].status == "refused"
    with pytest.raises(ValueError, match="must be empty"):
        subject.preflight_synthetic_targeted_capture(
            evidences=evidences, frame_ids=("a", "b", "c", "context"),
            project_path=tmp_path / "second.json", surface_support_sink=list(sink))


def test_entry_refusal_is_published_once_to_every_requested_sink(monkeypatch, tmp_path):
    evidences = tuple(_evidence(np.ones((4, 4), dtype=bool)) for _ in range(4))
    monkeypatch.setattr(subject.entry_mod, "build_image_derived_entry",
                       lambda *_a, **_k: SimpleNamespace(
                           graph=SimpleNamespace(status="unavailable"),
                           pose_graph=SimpleNamespace(status="unavailable"),
                           reason="no_connections", entry_sha256="4" * 64))
    public, private, support = [], [], []
    subject.preflight_synthetic_targeted_capture(
        evidences=evidences, frame_ids=("a", "b", "c", "context"),
        project_path=tmp_path / "all.json", evidence_bundle_sink=public,
        evidence_payload_sink=private, surface_support_sink=support,
        ownership_witness_payloads=(object(),))
    assert [len(public), len(private), len(support)] == [1, 1, 1]
    assert all(row[0].status == "refused" for row in (public, private, support))
