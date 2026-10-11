"""Frozen production policy for image-derived P1-D shift domains."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction.plane_shift_domain import ShiftDomainBudget

__all__ = [
    "IMAGE_DERIVED_SHIFT_DOMAIN_POLICY_SCHEMA",
    "IMAGE_DERIVED_SHIFT_DOMAIN_POLICY_PRODUCER",
    "IMAGE_DERIVED_SHIFT_DOMAIN_POLICY_VERSION",
    "ImageDerivedShiftDomainPolicy",
    "IMAGE_DERIVED_SHIFT_DOMAIN_POLICY",
]

IMAGE_DERIVED_SHIFT_DOMAIN_POLICY_SCHEMA = \
    "mc-photogrammetry-image-derived-shift-domain-policy-v3"
IMAGE_DERIVED_SHIFT_DOMAIN_POLICY_PRODUCER = \
    "m9-i-image-derived-shift-domain-policy-3"
IMAGE_DERIVED_SHIFT_DOMAIN_POLICY_VERSION = 3


def _payload(*, schema: str, producer: str, version: int,
             correlation_algorithm_id: str, max_shift_cells: int,
             budget: ShiftDomainBudget) -> dict:
    return {
        "schema": schema,
        "producer": producer,
        "version": version,
        "correlation_algorithm_id": correlation_algorithm_id,
        "max_shift_cells": max_shift_cells,
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
class ImageDerivedShiftDomainPolicy:
    """One immutable, replayable production policy; no field has a default."""

    schema: str
    producer: str
    version: int
    correlation_algorithm_id: str
    max_shift_cells: int
    budget: ShiftDomainBudget
    policy_sha256: str

    def __post_init__(self) -> None:
        if self.schema != IMAGE_DERIVED_SHIFT_DOMAIN_POLICY_SCHEMA:
            raise ValueError("unsupported image-derived shift-domain policy schema")
        if self.producer != IMAGE_DERIVED_SHIFT_DOMAIN_POLICY_PRODUCER:
            raise ValueError("unsupported image-derived shift-domain policy producer")
        if (type(self.version) is not int
                or self.version != IMAGE_DERIVED_SHIFT_DOMAIN_POLICY_VERSION):
            raise ValueError("unsupported image-derived shift-domain policy version")
        if self.correlation_algorithm_id != plane_graph.PLANE_CORRELATION_ALGORITHM:
            raise ValueError("unsupported plane-correlation algorithm")
        if type(self.max_shift_cells) is not int or self.max_shift_cells != 32:
            raise ValueError("max_shift_cells must be the exact v3 radius 32")
        if not isinstance(self.budget, ShiftDomainBudget):
            raise ValueError("policy requires an exact ShiftDomainBudget")
        try:
            replayed = ShiftDomainBudget(**self.budget.to_record())
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("shift-domain budget does not replay") from error
        if replayed.to_record() != self.budget.to_record():
            raise ValueError("shift-domain budget does not replay exactly")
        expected = _digest(_payload(
            schema=self.schema, producer=self.producer, version=self.version,
            correlation_algorithm_id=self.correlation_algorithm_id,
            max_shift_cells=self.max_shift_cells, budget=replayed))
        if not _is_sha256(self.policy_sha256) or self.policy_sha256 != expected:
            raise ValueError("image-derived shift-domain policy digest mismatch")

    def to_record(self) -> dict:
        record = _payload(
            schema=self.schema, producer=self.producer, version=self.version,
            correlation_algorithm_id=self.correlation_algorithm_id,
            max_shift_cells=self.max_shift_cells, budget=self.budget)
        record["policy_sha256"] = self.policy_sha256
        return record


def _builtin_policy() -> ImageDerivedShiftDomainPolicy:
    budget = ShiftDomainBudget(
        max_domains=10,
        max_padded_fft_cells=2_000_000,
        max_shift_positions=100_000,
        max_overlap_cell_visits=100_000_000,
        max_buffered_entries=100_000,
    )
    fields = {
        "schema": IMAGE_DERIVED_SHIFT_DOMAIN_POLICY_SCHEMA,
        "producer": IMAGE_DERIVED_SHIFT_DOMAIN_POLICY_PRODUCER,
        "version": IMAGE_DERIVED_SHIFT_DOMAIN_POLICY_VERSION,
        "correlation_algorithm_id": plane_graph.PLANE_CORRELATION_ALGORITHM,
        "max_shift_cells": 32,
        "budget": budget,
    }
    return ImageDerivedShiftDomainPolicy(
        **fields, policy_sha256=_digest(_payload(**fields)))


IMAGE_DERIVED_SHIFT_DOMAIN_POLICY = _builtin_policy()
"""The sole version-2 image-derived P1-D production policy."""
