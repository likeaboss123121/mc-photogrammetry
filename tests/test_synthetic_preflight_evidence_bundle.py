from types import SimpleNamespace
from dataclasses import replace

from mc_photogrammetry import evaluation_frozen_planes as truth_mod
from mc_photogrammetry import evaluation_signature_sampling as sampling_mod
from mc_photogrammetry import synthetic_targeted_capture_preflight as subject
from mc_photogrammetry.reconstruction import plane_graph
from test_synthetic_targeted_capture_preflight import _evidence
import numpy as np


def _rows():
    evidences = tuple(_evidence(np.ones((4, 4), dtype=bool)) for _ in range(2))
    frames = ("left", "right")
    ownership = tuple(truth_mod.RenderedTruthOwnershipBindingV1(
        "mc-photogrammetry-rendered-truth-owner-binding-v1",
        evidence.evidence_sha256, "a" * 64, "b" * 64, "c" * 64)
        for evidence in evidences)
    signatures = tuple(sampling_mod.SignatureSamplingBindingV1(
        sampling_mod.SCHEMA, frame, 0, 0, character * 64,
        evidence.evidence_sha256, "d" * 64)
        for frame, evidence, character in zip(frames, evidences, ("e", "f")))
    pairs = np.asarray([[0, 0, 0, 0]], dtype=np.int64); pairs.flags.writeable = False
    homography = np.eye(3); homography.flags.writeable = False
    matcher = plane_graph.MatcherPairWitnessV1(
        "mc-photogrammetry-matcher-pair-witness-v1",
        plane_graph.PLANE_CORRELATION_ALGORITHM, 32, 0,
        homography, "1" * 64, (132, 132), (0, 0), 1, pairs)
    envelope = plane_graph.GraphMatcherWitnessEnvelopeV1(
        "mc-photogrammetry-graph-matcher-witness-envelope-v1", "2" * 64,
        signatures[0].signature_sha256, signatures[1].signature_sha256,
        "left", "right", 0, 0, 0, 0, 0, (0, 0), 1, 4.0, .8,
        matcher.witness_sha256)
    traced = plane_graph.PlanePairTrace(
        left="left", right="right", left_axis=0, right_axis=0, quadrant=0,
        admitted=True, reason="", raw_max_shift_rc=None, raw_max_value=None,
        raw_max_support=None, supported_max_shift_rc=(0, 0),
        supported_max_value=1.0, supported_max_support=1,
        left_plane=0, right_plane=0, selected=True, psr=4.0, score=.8,
        overlap_cells=1, valid_cells=1, overlap_fraction=1.0,
        cell_shift_rc=(0, 0), left_structural_share=.5,
        right_structural_share=.5)
    trace = SimpleNamespace(pair_attribution=(traced,), attribution_sha256="3" * 64)
    return evidences, frames, ownership, signatures, (envelope,), trace


def test_complete_bundle_is_deterministic_and_digest_bound():
    values = _rows()
    first = subject._assemble_evidence_bundle(*values)
    second = subject._assemble_evidence_bundle(*values)
    assert isinstance(first, subject.SyntheticPreflightEvidenceBundleV1)
    assert first == second and first.bundle_sha256
    assert len(first.signature_binding_sha256) == 2
    assert len(first.matcher_envelope_sha256) == 1


def test_missing_duplicate_or_mismatched_join_refuses():
    evidences, frames, ownership, signatures, matcher, trace = _rows()
    missing = subject._assemble_evidence_bundle(
        evidences, frames, ownership[:-1], signatures, matcher, trace)
    assert missing.status == "refused"
    duplicate = subject._assemble_evidence_bundle(
        evidences, frames, ownership, signatures, matcher + matcher, trace)
    assert duplicate.status == "refused"
    bad_trace = SimpleNamespace(
        pair_attribution=(replace(trace.pair_attribution[0], overlap_cells=2),),
        attribution_sha256=trace.attribution_sha256)
    mismatch = subject._assemble_evidence_bundle(
        evidences, frames, ownership, signatures, matcher, bad_trace)
    assert mismatch.status == "refused"


def test_actual_preflight_publishes_typed_refusal_when_bundle_input_missing(
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
        project_path=tmp_path / "project.json", evidence_bundle_sink=sink)
    assert report.status == "insufficient_evidence"
    assert len(sink) == 1
    assert isinstance(sink[0], subject.SyntheticPreflightEvidenceRefusalV1)
    assert sink[0].stage == "ownership_bindings"

    bindings = tuple(truth_mod.RenderedTruthOwnershipBindingV1(
        "mc-photogrammetry-rendered-truth-owner-binding-v1",
        evidence.evidence_sha256, "a" * 64, "b" * 64, "c" * 64)
        for evidence in evidences)
    supplied = []
    second = subject.preflight_synthetic_targeted_capture(
        evidences=evidences, frame_ids=("a", "b", "c", "context"),
        project_path=tmp_path / "project-2.json", evidence_bundle_sink=supplied,
        ownership_bindings=bindings)
    assert second.to_record() == report.to_record()
    assert len(supplied) == 1 and supplied[0].stage == "entry"
