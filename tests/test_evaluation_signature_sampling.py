from dataclasses import replace

import numpy as np

from mc_photogrammetry import evaluation_signature_sampling as subject
from mc_photogrammetry.pipeline.image_derived_signatures import SignatureEvidence
from mc_photogrammetry.reconstruction import lattice, lattice_planes
from mc_photogrammetry.vision.gui_mask import create_gui_mask
from test_synthetic_targeted_capture_preflight import _evidence, _frozen


def _fixture():
    shape = (32, 32)
    usable = _frozen(np.logical_not(create_gui_mask(*shape)))
    rgb = np.indices(shape).sum(axis=0).astype(np.uint8)
    rgb = _frozen(np.repeat(rgb[..., None], 3, axis=2))
    truth = replace(_evidence(np.ones(shape, dtype=bool)), rgb=rgb,
                    usable=usable, evidence_sha256="")
    reading = lattice_planes.PlaneReading(
        frame="frame-a", axis=2, side=1.0,
        alternatives=(lattice_planes.DistanceAlternative(1.0, 1.0, 1.0),),
        period_pixels=8.0, phase_pixels=(0.0, 0.0), extent=1.0, samples=17,
        polygon=None, coverage_cells=4, structural_share=.5, confidence=.8,
        rectification=lattice_planes.PlaneRectification(
            tuple(tuple(float(v) for v in row) for row in np.eye(3)), 12.0,
            (16.0, 16.0), (0.0, 0.0)))
    signature = lattice_planes.plane_signature(rgb, reading, 1.0, usable=usable)
    signature = replace(signature, cells=_frozen(signature.cells),
                        interior=_frozen(signature.interior),
                        coverage=_frozen(signature.coverage))
    evidence = SignatureEvidence("frame-a", 0, 2, 0, 1.0, signature)
    return truth, evidence


def test_exact_signature_replay_emits_deterministic_binding():
    truth, signature = _fixture()
    first = subject.replay_signature_sampling_witness(
        signature_evidence=signature, truth_evidence=truth)
    second = subject.replay_signature_sampling_witness(
        signature_evidence=signature, truth_evidence=truth)
    assert isinstance(first, subject.SignatureSamplingBindingV1)
    assert first == second
    assert first.frame_id == "frame-a" and first.canonical_reading_index == 0
    assert first.signature_sha256 == signature.signature_sha256
    assert first.truth_evidence_sha256 == truth.evidence_sha256


def test_changed_signature_and_usable_mask_refuse_without_approximation():
    truth, signature = _fixture()
    cells = signature.signature.cells.copy(); cells[0, 0] += 1; cells.flags.writeable = False
    changed_signature = replace(
        signature, signature=replace(signature.signature, cells=cells),
        signature_sha256="")
    mismatch = subject.replay_signature_sampling_witness(
        signature_evidence=changed_signature, truth_evidence=truth)
    assert mismatch.reason == "replay_mismatch" and mismatch.stage == "signature"
    usable = truth.usable.copy(); usable[0, 0] = ~usable[0, 0]; usable.flags.writeable = False
    changed_truth = replace(truth, usable=usable, evidence_sha256="")
    refusal = subject.replay_signature_sampling_witness(
        signature_evidence=signature, truth_evidence=changed_truth)
    assert refusal.reason == "usable_mask_mismatch"


def test_sampling_budget_refusal_is_typed_and_atomic():
    truth, signature = _fixture()
    result = subject.replay_signature_sampling_witness(
        signature_evidence=signature, truth_evidence=truth,
        witness_budget=lattice.WitnessBudgetV1(
            max_rectified_samples=0, max_bilinear_taps=0,
            max_cell_memberships=0, max_payload_bytes=0))
    assert isinstance(result, subject.SignatureSamplingReplayRefusalV1)
    assert result.reason == "budget_exhausted"
    assert result.signature_sha256 == signature.signature_sha256


def test_binding_digest_tamper_refuses():
    truth, signature = _fixture()
    result = subject.replay_signature_sampling_witness(
        signature_evidence=signature, truth_evidence=truth)
    try:
        replace(result, witness_sha256="0" * 64)
    except ValueError as error:
        assert "digest mismatch" in str(error)
    else:
        raise AssertionError("tampered binding was accepted")
    malformed = subject.replay_signature_sampling_witness(
        signature_evidence=object(), truth_evidence=truth)
    assert malformed.reason == "provenance_mismatch"
