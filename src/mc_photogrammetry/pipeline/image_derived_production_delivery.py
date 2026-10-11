"""Normal-mode composition of image-derived production and verified delivery."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from mc_photogrammetry.pipeline import image_derived_delivery as delivery_mod
from mc_photogrammetry.pipeline import image_derived_entry as entry_mod
from mc_photogrammetry.pipeline import image_derived_output_delivery as output_delivery
from mc_photogrammetry.pipeline import image_derived_production as production_mod
from mc_photogrammetry.reconstruction import plane_placement

__all__ = ["IMAGE_DERIVED_PRODUCTION_DELIVERY_SCHEMA",
           "IMAGE_DERIVED_PRODUCTION_DELIVERY_PRODUCER",
           "ImageDerivedProductionDeliveryResult",
           "derive_and_deliver_image_output"]

IMAGE_DERIVED_PRODUCTION_DELIVERY_SCHEMA = \
    "mc-photogrammetry-image-derived-production-delivery"
IMAGE_DERIVED_PRODUCTION_DELIVERY_PRODUCER = \
    "r3-d-image-derived-production-delivery-1"


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")).hexdigest()


def _hex(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


@dataclass(frozen=True, slots=True)
class ImageDerivedProductionDeliveryResult:
    schema: str
    producer: str
    status: str
    reason: str
    production_sha256: str
    delivery_sha256: str
    production: production_mod.ImageDerivedProductionResult
    delivery: delivery_mod.DeliveryResult | delivery_mod.DeliveryRefusal | None
    result_sha256: str = ""

    def __post_init__(self) -> None:
        if (self.schema != IMAGE_DERIVED_PRODUCTION_DELIVERY_SCHEMA
                or self.producer != IMAGE_DERIVED_PRODUCTION_DELIVERY_PRODUCER
                or self.status not in (
                    "delivered", "geometry_unavailable", "review", "refused",
                    "budget_exhausted", "unavailable")
                or type(self.reason) is not str
                or not _hex(self.production_sha256)
                or not isinstance(self.production,
                                  production_mod.ImageDerivedProductionResult)):
            raise ValueError("production-delivery result is malformed")
        production = production_mod.ImageDerivedProductionResult(**{
            field.name: getattr(self.production, field.name)
            for field in dataclasses.fields(
                production_mod.ImageDerivedProductionResult)})
        if production.result_sha256 != self.production_sha256:
            raise ValueError("production-delivery source does not bind")
        if self.delivery is None:
            if self.delivery_sha256 or self.status == "delivered" \
                    or production.status == "promoted":
                raise ValueError("production-delivery terminal is inconsistent")
        elif isinstance(self.delivery, delivery_mod.DeliveryResult):
            replayed = delivery_mod.DeliveryResult(**{
                field.name: getattr(self.delivery, field.name)
                for field in dataclasses.fields(delivery_mod.DeliveryResult)})
            if (self.status != "delivered" or self.reason
                    or self.delivery_sha256 != replayed.artifact_sha256
                    or production.output is None
                    or replayed.promotion_sha256
                        != production.output.promotion.promotion_sha256):
                raise ValueError("delivered result does not bind promotion")
        elif isinstance(self.delivery, delivery_mod.DeliveryRefusal):
            replayed = delivery_mod.DeliveryRefusal(**{
                field.name: getattr(self.delivery, field.name)
                for field in dataclasses.fields(delivery_mod.DeliveryRefusal)})
            if (self.status != "refused" or not self.reason
                    or self.delivery_sha256 != replayed.refusal_sha256):
                raise ValueError("delivery refusal does not bind")
        else:
            raise ValueError("production-delivery carries a foreign delivery")
        payload = {
            "schema": self.schema, "producer": self.producer,
            "status": self.status, "reason": self.reason,
            "production_sha256": self.production_sha256,
            "delivery_sha256": self.delivery_sha256}
        digest = _sha(payload)
        if self.result_sha256 not in ("", digest):
            raise ValueError("production-delivery digest mismatch")
        object.__setattr__(self, "result_sha256", digest)


def _make(production, *, status, reason="", delivery=None):
    digest = ""
    if isinstance(delivery, delivery_mod.DeliveryResult):
        digest = delivery.artifact_sha256
    elif isinstance(delivery, delivery_mod.DeliveryRefusal):
        digest = delivery.refusal_sha256
    return ImageDerivedProductionDeliveryResult(
        IMAGE_DERIVED_PRODUCTION_DELIVERY_SCHEMA,
        IMAGE_DERIVED_PRODUCTION_DELIVERY_PRODUCER, status, reason,
        production.result_sha256, digest, production, delivery)


def derive_and_deliver_image_output(
    image_inputs: Iterable[str | Path], *, destination: Path, name: str,
    description: str, overwrite: bool = False, project_path: str | Path,
    recursive: bool = False,
    max_frames: int = entry_mod.DEFAULT_MAX_FRAMES,
    max_pairs: int = entry_mod.DEFAULT_MAX_PAIRS,
    max_total_support: int = entry_mod.DEFAULT_MAX_TOTAL_SUPPORT,
    max_edges: int = entry_mod.DEFAULT_MAX_EDGES,
    max_cycle_checks: int = entry_mod.DEFAULT_MAX_CYCLE_CHECKS,
    rotation_tolerance_degrees: float =
        entry_mod.DEFAULT_ROTATION_TOLERANCE_DEGREES,
    direction_tolerance: float = entry_mod.DEFAULT_DIRECTION_TOLERANCE,
) -> (ImageDerivedProductionDeliveryResult
      | plane_placement.PlacementRefusal):
    """Derive from ordinary images and deliver only an exact promoted result."""
    production = production_mod.derive_image_output_from_paths(
        image_inputs, project_path=project_path, recursive=recursive,
        max_frames=max_frames, max_pairs=max_pairs,
        max_total_support=max_total_support, max_edges=max_edges,
        max_cycle_checks=max_cycle_checks,
        rotation_tolerance_degrees=rotation_tolerance_degrees,
        direction_tolerance=direction_tolerance)
    if isinstance(production, plane_placement.PlacementRefusal):
        return production
    if production.status != "promoted":
        return _make(production, status=production.status,
                     reason=production.reason)
    delivered = output_delivery.deliver_image_derived_output(
        output_result=production.output, destination=destination, name=name,
        description=description, overwrite=overwrite, exporter=None)
    if isinstance(delivered, delivery_mod.DeliveryRefusal):
        return _make(production, status="refused", reason=delivered.reason,
                     delivery=delivered)
    return _make(production, status="delivered", delivery=delivered)
