"""Frozen production policy for image-derived V2-A endpoint provenance."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from mc_photogrammetry.reconstruction.plane_endpoint_provenance import V2ABudget

__all__ = [
    "IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY_SCHEMA",
    "IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY_PRODUCER",
    "IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY_VERSION",
    "ImageDerivedEndpointProvenancePolicy",
    "IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY",
]

IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY_SCHEMA = \
    "mc-photogrammetry-image-derived-endpoint-provenance-policy"
IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY_PRODUCER = \
    "m9-o1-image-derived-endpoint-provenance-policy-1"
IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY_VERSION = 1


def _payload(*, schema: str, producer: str, version: int,
             budget: V2ABudget) -> dict:
    return {
        "schema": schema,
        "producer": producer,
        "version": version,
        "budget": budget.to_record(),
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
class ImageDerivedEndpointProvenancePolicy:
    """One immutable, replayable V2-A production policy."""

    schema: str
    producer: str
    version: int
    budget: V2ABudget
    policy_sha256: str

    def __post_init__(self) -> None:
        if self.schema != IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY_SCHEMA:
            raise ValueError("unsupported endpoint-provenance policy schema")
        if self.producer != IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY_PRODUCER:
            raise ValueError("unsupported endpoint-provenance policy producer")
        if (type(self.version) is not int
                or self.version != IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY_VERSION):
            raise ValueError("unsupported endpoint-provenance policy version")
        if not isinstance(self.budget, V2ABudget):
            raise ValueError("policy requires an exact V2ABudget")
        try:
            replayed = V2ABudget(**self.budget.to_record())
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("endpoint-provenance budget does not replay") from error
        if replayed.to_record() != self.budget.to_record():
            raise ValueError("endpoint-provenance budget does not replay exactly")
        expected = _digest(_payload(
            schema=self.schema, producer=self.producer, version=self.version,
            budget=replayed))
        if not _is_sha256(self.policy_sha256) or self.policy_sha256 != expected:
            raise ValueError("endpoint-provenance policy digest mismatch")

    def to_record(self) -> dict:
        record = _payload(
            schema=self.schema, producer=self.producer, version=self.version,
            budget=self.budget)
        record["policy_sha256"] = self.policy_sha256
        return record


def _builtin_policy() -> ImageDerivedEndpointProvenancePolicy:
    budget = V2ABudget(
        max_readings=20,
        max_edges=10,
        max_endpoint_labels=10_000_000,
        max_residues_inspected=40_000_000,
        max_edge_candidates_inspected=10_000_000,
        max_partial_label_assignments=10_000_000,
        max_consistency_checks=10_000_000,
        max_result_validation_cross_links=20_000_000,
        max_distinct_assignments=2,
        max_retained_witnesses=2,
    )
    fields = {
        "schema": IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY_SCHEMA,
        "producer": IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY_PRODUCER,
        "version": IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY_VERSION,
        "budget": budget,
    }
    return ImageDerivedEndpointProvenancePolicy(
        **fields, policy_sha256=_digest(_payload(**fields)))


IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY = _builtin_policy()
"""The sole version-1 image-derived V2-A production policy."""
