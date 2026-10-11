"""Frozen production policy for image-derived aligned V1 supports."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from mc_photogrammetry.reconstruction import plane_region_support

__all__ = [
    "IMAGE_DERIVED_REGION_SUPPORT_POLICY_SCHEMA",
    "IMAGE_DERIVED_REGION_SUPPORT_POLICY_PRODUCER",
    "IMAGE_DERIVED_REGION_SUPPORT_POLICY_VERSION",
    "ImageDerivedRegionSupportPolicy",
    "IMAGE_DERIVED_REGION_SUPPORT_POLICY",
]

IMAGE_DERIVED_REGION_SUPPORT_POLICY_SCHEMA = \
    "mc-photogrammetry-image-derived-region-support-policy"
IMAGE_DERIVED_REGION_SUPPORT_POLICY_PRODUCER = \
    "m9-q1-image-derived-region-support-policy-1"
IMAGE_DERIVED_REGION_SUPPORT_POLICY_VERSION = 1


def _payload(*, schema: str, producer: str, version: int,
             budget: plane_region_support.AlignedSupportBudget) -> dict:
    return {"schema": schema, "producer": producer, "version": version,
            "budget": budget.to_record()}


def _digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


@dataclass(frozen=True, slots=True)
class ImageDerivedRegionSupportPolicy:
    schema: str
    producer: str
    version: int
    budget: plane_region_support.AlignedSupportBudget
    policy_sha256: str

    def __post_init__(self) -> None:
        if self.schema != IMAGE_DERIVED_REGION_SUPPORT_POLICY_SCHEMA:
            raise ValueError("unsupported region-support policy schema")
        if self.producer != IMAGE_DERIVED_REGION_SUPPORT_POLICY_PRODUCER:
            raise ValueError("unsupported region-support policy producer")
        if (type(self.version) is not int
                or self.version != IMAGE_DERIVED_REGION_SUPPORT_POLICY_VERSION):
            raise ValueError("unsupported region-support policy version")
        if not isinstance(self.budget, plane_region_support.AlignedSupportBudget):
            raise ValueError("policy requires an exact AlignedSupportBudget")
        record = self.budget.to_record()
        caps = {key: value for key, value in record.items()
                if key != "budget_sha256"}
        try:
            replayed = plane_region_support.make_support_budget(**caps)
        except (TypeError, ValueError) as error:
            raise ValueError("region-support budget does not replay") from error
        if replayed.to_record() != record:
            raise ValueError("region-support budget does not replay exactly")
        expected = _digest(_payload(
            schema=self.schema, producer=self.producer, version=self.version,
            budget=replayed))
        if not _is_sha256(self.policy_sha256) or self.policy_sha256 != expected:
            raise ValueError("region-support policy digest mismatch")

    def to_record(self) -> dict:
        record = _payload(schema=self.schema, producer=self.producer,
                          version=self.version, budget=self.budget)
        record["policy_sha256"] = self.policy_sha256
        return record


def _builtin_policy() -> ImageDerivedRegionSupportPolicy:
    budget = plane_region_support.make_support_budget(
        max_cells=10_000_000,
        max_region_pixel_tests=100_000_000,
        max_usable_pixel_tests=100_000_000,
        max_polygon_vertices=100_000_000,
        max_supported_cells=10_000_000)
    fields = {"schema": IMAGE_DERIVED_REGION_SUPPORT_POLICY_SCHEMA,
              "producer": IMAGE_DERIVED_REGION_SUPPORT_POLICY_PRODUCER,
              "version": IMAGE_DERIVED_REGION_SUPPORT_POLICY_VERSION,
              "budget": budget}
    return ImageDerivedRegionSupportPolicy(
        **fields, policy_sha256=_digest(_payload(**fields)))


IMAGE_DERIVED_REGION_SUPPORT_POLICY = _builtin_policy()
