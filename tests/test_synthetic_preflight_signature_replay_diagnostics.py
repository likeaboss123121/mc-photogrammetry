from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mc_photogrammetry import evaluation_signature_sampling as sampling
from mc_photogrammetry import synthetic_targeted_capture_preflight as subject
from mc_photogrammetry.reconstruction import lattice
from test_evaluation_signature_sampling import _fixture
from test_synthetic_targeted_capture_preflight import _evidence


def test_complete_and_refused_rows_preserve_identity_and_order():
    truth, first = _fixture()
    second = replace(first, canonical_index=1, alternative_index=2,
                     signature_sha256="")
    bindings, payloads, rows = subject._collect_signature_replays(
        (second, first), {"frame-a": truth}, None, collect_payload=False)
    assert payloads == []
    assert [(row.canonical_index, row.alternative_index) for row in rows] == [
        (1, 2), (0, 0)]
    assert all(isinstance(row.result, sampling.SignatureSamplingBindingV1)
               for row in rows)
    assert tuple(row.result for row in rows) == tuple(bindings)

    budget = lattice.WitnessBudgetV1(
        max_rectified_samples=0, max_bilinear_taps=0,
        max_cell_memberships=0, max_payload_bytes=0)
    refused, _, refused_rows = subject._collect_signature_replays(
        (first,), {"frame-a": truth}, budget, collect_payload=False)
    assert refused_rows[0].result is refused[0]
    assert refused_rows[0].result.reason == "budget_exhausted"
    assert refused_rows[0].result.stage
    assert "limit=" in refused_rows[0].result.detail


def test_diagnostic_row_rejects_malformed_or_crossed_identity():
    truth, signature = _fixture()
    result = sampling.replay_signature_sampling_witness(
        signature_evidence=signature, truth_evidence=truth)
    with pytest.raises(ValueError, match="identity mismatch"):
        subject.SignatureReplayDiagnosticRowV1(
            "other", 0, 0, signature.signature_sha256, result)
    with pytest.raises(ValueError, match="malformed"):
        subject.SignatureReplayDiagnosticRowV1(
            "frame-a", 0, 0, "0" * 64, result)


def test_batch_publication_is_atomic_when_later_identity_is_missing():
    truth, signature = _fixture()
    later = SimpleNamespace(
        frame_id="missing", canonical_index=1, alternative_index=0,
        signature_sha256="0" * 64)
    sink = []
    with pytest.raises(KeyError):
        rows = subject._collect_signature_replays(
            (signature, later), {"frame-a": truth}, None,
            collect_payload=False)[2]
        sink.extend(rows)
    assert sink == []


def test_sink_off_is_inert_and_malformed_sink_fails_before_pipeline(
        monkeypatch, tmp_path):
    evidences = tuple(_evidence(np.ones((4, 4), dtype=bool)) for _ in range(4))
    calls = []
    monkeypatch.setattr(subject.sampling_mod, "replay_signature_sampling_witness",
                        lambda **_kwargs: calls.append(1))
    monkeypatch.setattr(subject.entry_mod, "build_image_derived_entry",
                        lambda *_a, **_k: SimpleNamespace(
                            graph=SimpleNamespace(status="unavailable"),
                            pose_graph=SimpleNamespace(status="unavailable"),
                            reason="no_connections", entry_sha256="4" * 64))
    report = subject.preflight_synthetic_targeted_capture(
        evidences=evidences, frame_ids=("a", "b", "c", "context"),
        project_path=tmp_path / "off.json")
    assert report.status == "insufficient_evidence" and calls == []
    with pytest.raises(ValueError, match="diagnostic sink must be empty"):
        subject.preflight_synthetic_targeted_capture(
            evidences=evidences, frame_ids=("a", "b", "c", "context"),
            project_path=tmp_path / "bad.json",
            signature_replay_diagnostic_sink=[object()])
    with pytest.raises(ValueError, match="diagnostic sink must be empty"):
        subject.preflight_synthetic_targeted_capture(
            evidences=evidences, frame_ids=("a", "b", "c", "context"),
            project_path=tmp_path / "bad-type.json",
            signature_replay_diagnostic_sink=())
