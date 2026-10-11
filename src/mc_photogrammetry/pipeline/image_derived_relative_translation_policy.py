"""Frozen production policy for image-derived C2-R-B closure solving."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from mc_photogrammetry.reconstruction.plane_relative_translation import C2RBudget

__all__ = [
    "IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY_SCHEMA",
    "IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY_PRODUCER",
    "IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY_VERSION",
    "ImageDerivedRelativeTranslationPolicy",
    "IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY",
]

IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY_SCHEMA = \
    "mc-photogrammetry-image-derived-relative-translation-policy"
IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY_PRODUCER = \
    "m9-n1-image-derived-relative-translation-policy-1"
IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY_VERSION = 1


def _payload(*, schema: str, producer: str, version: int,
             comparison_tolerance: float, closure_tolerance: float,
             budget: C2RBudget) -> dict:
    return {
        "schema": schema,
        "producer": producer,
        "version": version,
        "comparison_tolerance": comparison_tolerance,
        "closure_tolerance": closure_tolerance,
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
class ImageDerivedRelativeTranslationPolicy:
    """One immutable, replayable C2-R-B production policy."""

    schema: str
    producer: str
    version: int
    comparison_tolerance: float
    closure_tolerance: float
    budget: C2RBudget
    policy_sha256: str

    def __post_init__(self) -> None:
        if self.schema != IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY_SCHEMA:
            raise ValueError("unsupported relative-translation policy schema")
        if self.producer != IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY_PRODUCER:
            raise ValueError("unsupported relative-translation policy producer")
        if (type(self.version) is not int
                or self.version != IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY_VERSION):
            raise ValueError("unsupported relative-translation policy version")
        for name in ("comparison_tolerance", "closure_tolerance"):
            value = getattr(self, name)
            if (type(value) is not float or not math.isfinite(value)
                    or value < 0.0):
                raise ValueError(f"{name} must be a finite non-negative float")
        if not isinstance(self.budget, C2RBudget):
            raise ValueError("policy requires an exact C2RBudget")
        try:
            replayed = C2RBudget(**self.budget.to_record())
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("relative-translation budget does not replay") from error
        if replayed.to_record() != self.budget.to_record():
            raise ValueError("relative-translation budget does not replay exactly")
        expected = _digest(_payload(
            schema=self.schema, producer=self.producer, version=self.version,
            comparison_tolerance=self.comparison_tolerance,
            closure_tolerance=self.closure_tolerance, budget=replayed))
        if not _is_sha256(self.policy_sha256) or self.policy_sha256 != expected:
            raise ValueError("relative-translation policy digest mismatch")

    def to_record(self) -> dict:
        record = _payload(
            schema=self.schema, producer=self.producer, version=self.version,
            comparison_tolerance=self.comparison_tolerance,
            closure_tolerance=self.closure_tolerance, budget=self.budget)
        record["policy_sha256"] = self.policy_sha256
        return record


def _builtin_policy() -> ImageDerivedRelativeTranslationPolicy:
    budget = C2RBudget(
        max_components=10,
        max_frames=32,
        max_edges=10,
        max_tree_edges=10,
        max_chords=10,
        max_cycle_rows=30,
        max_p1d_entries=2_000_000,
        max_residue_values=10_000_000,
        max_candidate_views_scanned=10_000_000,
        max_partial_assignments=10_000_000,
        max_closure_component_checks=10_000_000,
        max_membership_probes=10_000_000,
        max_memoized_terms=10_000_000,
        max_provenance_witnesses=10_000_000,
        max_certificates=100_000,
    )
    fields = {
        "schema": IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY_SCHEMA,
        "producer": IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY_PRODUCER,
        "version": IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY_VERSION,
        "comparison_tolerance": 1e-12,
        "closure_tolerance": 0.0,
        "budget": budget,
    }
    return ImageDerivedRelativeTranslationPolicy(
        **fields, policy_sha256=_digest(_payload(**fields)))


IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY = _builtin_policy()
"""The sole version-1 image-derived C2-R-B production policy."""
