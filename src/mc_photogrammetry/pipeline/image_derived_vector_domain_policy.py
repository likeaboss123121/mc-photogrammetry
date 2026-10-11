"""Frozen production policy for image-derived C2-R vector domains."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from mc_photogrammetry.reconstruction.plane_vector_domain import C2RVectorBudget

__all__ = [
    "IMAGE_DERIVED_VECTOR_DOMAIN_POLICY_SCHEMA",
    "IMAGE_DERIVED_VECTOR_DOMAIN_POLICY_PRODUCER",
    "IMAGE_DERIVED_VECTOR_DOMAIN_POLICY_VERSION",
    "ImageDerivedVectorDomainPolicy",
    "IMAGE_DERIVED_VECTOR_DOMAIN_POLICY",
]

IMAGE_DERIVED_VECTOR_DOMAIN_POLICY_SCHEMA = \
    "mc-photogrammetry-image-derived-vector-domain-policy"
IMAGE_DERIVED_VECTOR_DOMAIN_POLICY_PRODUCER = \
    "m9-m1-image-derived-vector-domain-policy-1"
IMAGE_DERIVED_VECTOR_DOMAIN_POLICY_VERSION = 1


def _payload(*, schema: str, producer: str, version: int,
             comparison_tolerance: float, budget: C2RVectorBudget) -> dict:
    return {
        "schema": schema,
        "producer": producer,
        "version": version,
        "comparison_tolerance": comparison_tolerance,
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
class ImageDerivedVectorDomainPolicy:
    """One immutable, replayable production policy; no field has a default."""

    schema: str
    producer: str
    version: int
    comparison_tolerance: float
    budget: C2RVectorBudget
    policy_sha256: str

    def __post_init__(self) -> None:
        if self.schema != IMAGE_DERIVED_VECTOR_DOMAIN_POLICY_SCHEMA:
            raise ValueError("unsupported image-derived vector-domain policy schema")
        if self.producer != IMAGE_DERIVED_VECTOR_DOMAIN_POLICY_PRODUCER:
            raise ValueError("unsupported image-derived vector-domain policy producer")
        if (type(self.version) is not int
                or self.version != IMAGE_DERIVED_VECTOR_DOMAIN_POLICY_VERSION):
            raise ValueError("unsupported image-derived vector-domain policy version")
        if (type(self.comparison_tolerance) is not float
                or not math.isfinite(self.comparison_tolerance)
                or self.comparison_tolerance < 0.0):
            raise ValueError("comparison_tolerance must be a finite non-negative float")
        if not isinstance(self.budget, C2RVectorBudget):
            raise ValueError("policy requires an exact C2RVectorBudget")
        try:
            replayed = C2RVectorBudget(**self.budget.to_record())
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("vector-domain budget does not replay") from error
        if replayed.to_record() != self.budget.to_record():
            raise ValueError("vector-domain budget does not replay exactly")
        expected = _digest(_payload(
            schema=self.schema, producer=self.producer, version=self.version,
            comparison_tolerance=self.comparison_tolerance, budget=replayed))
        if not _is_sha256(self.policy_sha256) or self.policy_sha256 != expected:
            raise ValueError("image-derived vector-domain policy digest mismatch")

    def to_record(self) -> dict:
        record = _payload(
            schema=self.schema, producer=self.producer, version=self.version,
            comparison_tolerance=self.comparison_tolerance, budget=self.budget)
        record["policy_sha256"] = self.policy_sha256
        return record


def _builtin_policy() -> ImageDerivedVectorDomainPolicy:
    budget = C2RVectorBudget(
        max_requested_match_identities=10,
        max_joint_entries=2_000_000,
        max_axis_residue_pairs=10_000_000,
        max_vector_candidates=10_000_000,
        max_provenance_references=100_000_000,
    )
    fields = {
        "schema": IMAGE_DERIVED_VECTOR_DOMAIN_POLICY_SCHEMA,
        "producer": IMAGE_DERIVED_VECTOR_DOMAIN_POLICY_PRODUCER,
        "version": IMAGE_DERIVED_VECTOR_DOMAIN_POLICY_VERSION,
        "comparison_tolerance": 1e-12,
        "budget": budget,
    }
    return ImageDerivedVectorDomainPolicy(
        **fields, policy_sha256=_digest(_payload(**fields)))


IMAGE_DERIVED_VECTOR_DOMAIN_POLICY = _builtin_policy()
"""The sole version-1 image-derived C2-R production policy."""
