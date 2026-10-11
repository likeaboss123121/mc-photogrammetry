"""Frozen production policy for image-derived V2-B phase anchoring."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

from mc_photogrammetry.reconstruction import plane_phase_anchor

__all__ = [
    "IMAGE_DERIVED_PHASE_ANCHOR_POLICY_SCHEMA",
    "IMAGE_DERIVED_PHASE_ANCHOR_POLICY_PRODUCER",
    "IMAGE_DERIVED_PHASE_ANCHOR_POLICY_VERSION",
    "ImageDerivedPhaseAnchorPolicy",
    "IMAGE_DERIVED_PHASE_ANCHOR_POLICY",
]

IMAGE_DERIVED_PHASE_ANCHOR_POLICY_SCHEMA = \
    "mc-photogrammetry-image-derived-phase-anchor-policy"
IMAGE_DERIVED_PHASE_ANCHOR_POLICY_PRODUCER = \
    "m9-p1-image-derived-phase-anchor-policy-1"
IMAGE_DERIVED_PHASE_ANCHOR_POLICY_VERSION = 1


def _payload(*, schema: str, producer: str, version: int,
             budget: plane_phase_anchor.V2BAnchorBudget,
             calibration_sha256: str) -> dict:
    return {
        "schema": schema,
        "producer": producer,
        "version": version,
        "budget": budget.to_record(),
        "calibration_sha256": calibration_sha256,
    }


def _digest(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


@dataclass(frozen=True, slots=True)
class ImageDerivedPhaseAnchorPolicy:
    """One immutable, replayable V2-B production policy."""

    schema: str
    producer: str
    version: int
    budget: plane_phase_anchor.V2BAnchorBudget
    calibration_sha256: str
    policy_sha256: str

    def __post_init__(self) -> None:
        if self.schema != IMAGE_DERIVED_PHASE_ANCHOR_POLICY_SCHEMA:
            raise ValueError("unsupported phase-anchor policy schema")
        if self.producer != IMAGE_DERIVED_PHASE_ANCHOR_POLICY_PRODUCER:
            raise ValueError("unsupported phase-anchor policy producer")
        if (type(self.version) is not int
                or self.version != IMAGE_DERIVED_PHASE_ANCHOR_POLICY_VERSION):
            raise ValueError("unsupported phase-anchor policy version")
        if not isinstance(self.budget, plane_phase_anchor.V2BAnchorBudget):
            raise ValueError("policy requires an exact V2BAnchorBudget")
        try:
            record = self.budget.to_record()
            caps = {key: value for key, value in record.items()
                    if key != "budget_sha256"}
            replayed = plane_phase_anchor.make_anchor_budget(**caps)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("phase-anchor budget does not replay") from error
        if replayed.to_record() != record:
            raise ValueError("phase-anchor budget does not replay exactly")
        calibration = plane_phase_anchor.V2BAnchorCalibration(**{
            field.name: getattr(plane_phase_anchor.V2B_BUILTIN_CALIBRATION,
                                field.name)
            for field in dataclasses.fields(
                plane_phase_anchor.V2BAnchorCalibration)})
        if (not _is_sha256(self.calibration_sha256)
                or self.calibration_sha256 != calibration.calibration_sha256):
            raise ValueError("phase-anchor calibration digest mismatch")
        expected = _digest(_payload(
            schema=self.schema, producer=self.producer, version=self.version,
            budget=replayed, calibration_sha256=self.calibration_sha256))
        if not _is_sha256(self.policy_sha256) or self.policy_sha256 != expected:
            raise ValueError("phase-anchor policy digest mismatch")

    def to_record(self) -> dict:
        record = _payload(
            schema=self.schema, producer=self.producer, version=self.version,
            budget=self.budget, calibration_sha256=self.calibration_sha256)
        record["policy_sha256"] = self.policy_sha256
        return record


def _builtin_policy() -> ImageDerivedPhaseAnchorPolicy:
    budget = plane_phase_anchor.make_anchor_budget(
        max_readings=20,
        max_observations=10_000_000,
        max_frames=32,
        max_hypotheses_per_axis=10_000_000,
        max_hypotheses_total=30_000_000,
        max_contributor_refs_per_hypothesis=20,
        max_contributor_refs_total=200_000_000,
        max_source_records=20,
        max_boundary_interior_terms=20_000_000,
        max_d_profile_terms=10_000_000,
        max_payload_bytes=100_000_000,
        max_replay_crosslinks=40,
        max_pixel_samples=10_000_000,
        max_held_out_samples_per_reading=1_000_000,
        max_split_reconstructions=20,
        max_accepted_frame_supports=10_000_000,
    )
    fields = {
        "schema": IMAGE_DERIVED_PHASE_ANCHOR_POLICY_SCHEMA,
        "producer": IMAGE_DERIVED_PHASE_ANCHOR_POLICY_PRODUCER,
        "version": IMAGE_DERIVED_PHASE_ANCHOR_POLICY_VERSION,
        "budget": budget,
        "calibration_sha256":
            plane_phase_anchor.V2B_BUILTIN_CALIBRATION.calibration_sha256,
    }
    return ImageDerivedPhaseAnchorPolicy(
        **fields, policy_sha256=_digest(_payload(**fields)))


IMAGE_DERIVED_PHASE_ANCHOR_POLICY = _builtin_policy()
"""The sole version-1 image-derived V2-B production policy."""
