"""Thin verified-delivery gate for accepted image-derived output."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from mc_photogrammetry.pipeline import image_derived_delivery as delivery
from mc_photogrammetry.pipeline import image_derived_output as output_mod
from mc_photogrammetry.reconstruction import state_promotion as promotion

__all__ = ["deliver_image_derived_output"]

_ERRORS = (AttributeError, TypeError, ValueError, IndexError, KeyError,
           OverflowError, ZeroDivisionError)


def _refuse(reason: str, field: str, detail: object = ""):
    evidence = {"field": field}
    if detail:
        evidence["detail"] = str(detail)
    return delivery._refuse(reason, evidence)


def _replay(value: object):
    if not isinstance(value, output_mod.ImageDerivedOutputResult):
        return _refuse("invalid_input", "output_result")
    try:
        return output_mod.ImageDerivedOutputResult(**{
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(output_mod.ImageDerivedOutputResult)})
    except _ERRORS as error:
        return _refuse("invalid_input", "output_result", type(error).__name__)


def _deliver_promoted(
    promotion_result: promotion.PromotionResult,
    *, destination: Path, name: str, description: str,
    overwrite: bool, exporter: Any,
):
    if promotion_result.export_ready is not True:
        return _refuse("promotion_not_export_ready", "promotion.export_ready")
    if promotion_result.exact_state_eligible is not True:
        return _refuse("promotion_not_exact_state_eligible",
                       "promotion.exact_state_eligible")
    if (type(promotion_result.omitted_unknown) is not int
            or promotion_result.omitted_unknown != 0):
        return _refuse("promotion_omitted_unknown",
                       "promotion.omitted_unknown")
    return delivery.deliver_image_derived_litematic(
        promotion_result, destination, name=name, description=description,
        overwrite=overwrite, exporter=exporter)


def deliver_image_derived_output(
    *, output_result: object, destination: Path, name: str,
    description: str, overwrite: bool = False, exporter: Any = None,
) -> delivery.DeliveryResult | delivery.DeliveryRefusal:
    """Replay one promoted R2-A output and delegate verified delivery once."""
    replayed = _replay(output_result)
    if isinstance(replayed, delivery.DeliveryRefusal):
        return replayed
    if replayed.status != "promoted":
        return _refuse("invalid_promotion", "output_result.status",
                       replayed.status)
    if not isinstance(replayed.promotion, promotion.PromotionResult):
        return _refuse("invalid_promotion", "output_result.promotion")
    return _deliver_promoted(
        replayed.promotion, destination=destination, name=name,
        description=description, overwrite=overwrite, exporter=exporter)
