"""Evaluation-only exact replay binding for signature sampling witnesses."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from mc_photogrammetry import evaluation_frozen_planes as truth_mod
from mc_photogrammetry.pipeline import image_derived_signatures as signature_mod
from mc_photogrammetry.reconstruction import lattice, lattice_planes
from mc_photogrammetry.vision.gui_mask import create_gui_mask

SCHEMA = "mc-photogrammetry-signature-sampling-binding-v1"


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SignatureSamplingBindingV1:
    schema: str
    frame_id: str
    canonical_reading_index: int
    alternative_index: int
    signature_sha256: str
    truth_evidence_sha256: str
    witness_sha256: str
    report_sha256: str = ""

    def __post_init__(self) -> None:
        if (self.schema != SCHEMA or not self.frame_id
                or self.canonical_reading_index < 0 or self.alternative_index < 0
                or any(type(value) is not str or len(value) != 64 for value in
                       (self.signature_sha256, self.truth_evidence_sha256,
                        self.witness_sha256))):
            raise ValueError("signature sampling binding is malformed")
        digest = _sha(self.to_record(include_digest=False))
        if self.report_sha256 not in ("", digest):
            raise ValueError("signature sampling binding digest mismatch")
        object.__setattr__(self, "report_sha256", digest)

    def to_record(self, *, include_digest=True):
        result = {"schema": self.schema, "frame_id": self.frame_id,
                  "canonical_reading_index": self.canonical_reading_index,
                  "alternative_index": self.alternative_index,
                  "signature_sha256": self.signature_sha256,
                  "truth_evidence_sha256": self.truth_evidence_sha256,
                  "witness_sha256": self.witness_sha256}
        if include_digest:
            result["report_sha256"] = self.report_sha256
        return result


@dataclass(frozen=True, slots=True)
class SignatureSamplingWitnessPayloadV1:
    """Private carrier retaining the exact sampling witness beside its binding."""
    binding: SignatureSamplingBindingV1
    witness: lattice_planes.SignatureSamplingWitnessV1

    def __post_init__(self) -> None:
        if (not isinstance(self.binding, SignatureSamplingBindingV1)
                or not isinstance(self.witness, lattice_planes.SignatureSamplingWitnessV1)):
            raise ValueError("signature sampling payload is malformed")
        binding = SignatureSamplingBindingV1(**{
            name: getattr(self.binding, name)
            for name in self.binding.__dataclass_fields__})
        replay = lattice_planes.SignatureSamplingWitnessV1(**{
            name: getattr(self.witness, name)
            for name in self.witness.__dataclass_fields__})
        for nested in (self.witness.rgb_rectification,
                       self.witness.coverage_rectification,
                       self.witness.region_rectification):
            if nested is not None:
                lattice.RectificationWitnessV1(**{
                    name: getattr(nested, name)
                    for name in nested.__dataclass_fields__})
        if replay.witness_sha256 != binding.witness_sha256:
            raise ValueError("signature sampling payload digest mismatch")


@dataclass(frozen=True, slots=True)
class SignatureSamplingReplayRefusalV1:
    schema: str
    status: str
    reason: str
    stage: str
    detail: str
    signature_sha256: str
    truth_evidence_sha256: str
    report_sha256: str = ""

    def __post_init__(self) -> None:
        if (self.schema != SCHEMA or self.status != "refused"
                or self.reason not in ("provenance_mismatch", "replay_mismatch",
                                       "usable_mask_mismatch", "budget_exhausted")):
            raise ValueError("signature sampling refusal is malformed")
        digest = _sha(self.to_record(include_digest=False))
        if self.report_sha256 not in ("", digest):
            raise ValueError("signature sampling refusal digest mismatch")
        object.__setattr__(self, "report_sha256", digest)

    def to_record(self, *, include_digest=True):
        result = {"schema": self.schema, "status": self.status,
                  "reason": self.reason, "stage": self.stage,
                  "detail": self.detail,
                  "signature_sha256": self.signature_sha256,
                  "truth_evidence_sha256": self.truth_evidence_sha256}
        if include_digest:
            result["report_sha256"] = self.report_sha256
        return result


def _refusal(reason, stage, detail, signature="", truth=""):
    return SignatureSamplingReplayRefusalV1(
        SCHEMA, "refused", reason, stage, detail, signature, truth)


def replay_signature_sampling_witness(
    *, signature_evidence: signature_mod.SignatureEvidence,
    truth_evidence: truth_mod.RenderedTruthEvidence,
    witness_budget: lattice.WitnessBudgetV1 | None = None,
    witness_payload_sink=None,
) -> SignatureSamplingBindingV1 | SignatureSamplingReplayRefusalV1:
    """Replay one accepted signature and bind its exact sampling witness."""
    if witness_payload_sink is not None and len(witness_payload_sink):
        raise ValueError("signature witness payload sink must be empty")
    signature_sha = getattr(signature_evidence, "signature_sha256", "")
    truth_sha = getattr(truth_evidence, "evidence_sha256", "")
    try:
        if (not isinstance(signature_evidence, signature_mod.SignatureEvidence)
                or not isinstance(truth_evidence, truth_mod.RenderedTruthEvidence)):
            raise ValueError("unsupported replay input type")
        signature_evidence = signature_mod.SignatureEvidence(**{
            name: getattr(signature_evidence, name)
            for name in signature_evidence.__dataclass_fields__})
        expected_usable = np.logical_not(create_gui_mask(*truth_evidence.usable.shape))
        if not np.array_equal(expected_usable, truth_evidence.usable):
            return _refusal("usable_mask_mismatch", "usable_mask",
                            "truth usable mask differs from the production GUI mask",
                            signature_sha, truth_sha)
        sink = []
        replay = lattice_planes.plane_signature(
            truth_evidence.rgb, signature_evidence.signature.reading,
            signature_evidence.blocks_per_period, usable=truth_evidence.usable,
            witness_sink=sink, witness_budget=witness_budget)
        if len(sink) != 1:
            return _refusal("replay_mismatch", "witness",
                            "sampling witness publication is incomplete",
                            signature_sha, truth_sha)
        if isinstance(sink[0], lattice.WitnessRefusalV1):
            refusal = sink[0]
            return _refusal("budget_exhausted", refusal.stage,
                            f"limit={refusal.limit};planned={refusal.planned};actual={refusal.actual}",
                            signature_sha, truth_sha)
        expected = signature_evidence.signature
        equal = (replay.reading == expected.reading
                 and replay.blocks_per_period == expected.blocks_per_period
                 and replay.origin_uv == expected.origin_uv
                 and replay.region_footprint == expected.region_footprint
                 and np.array_equal(replay.cells, expected.cells)
                 and np.array_equal(replay.interior, expected.interior)
                 and np.array_equal(replay.coverage, expected.coverage))
        if not equal:
            return _refusal("replay_mismatch", "signature",
                            "replayed signature differs from accepted evidence",
                            signature_sha, truth_sha)
        binding = SignatureSamplingBindingV1(
            SCHEMA, signature_evidence.frame_id,
            signature_evidence.canonical_index,
            signature_evidence.alternative_index, signature_sha, truth_sha,
            sink[0].witness_sha256)
        if witness_payload_sink is not None:
            witness_payload_sink.append(
                SignatureSamplingWitnessPayloadV1(binding, sink[0]))
        return binding
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        return _refusal("provenance_mismatch", "input", type(error).__name__,
                        signature_sha, truth_sha)


__all__ = ["SignatureSamplingBindingV1", "SignatureSamplingWitnessPayloadV1",
           "SignatureSamplingReplayRefusalV1",
           "replay_signature_sampling_witness"]
