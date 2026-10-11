"""Derive V2-A endpoint provenance from accepted image-derived closure."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

from mc_photogrammetry.pipeline import image_derived_axis_support as axis_mod
from mc_photogrammetry.pipeline import image_derived_endpoint_provenance_policy as policy_mod
from mc_photogrammetry.pipeline import image_derived_relative_translation_policy as relative_policy_mod
from mc_photogrammetry.pipeline import image_derived_relative_translations as relative_mod
from mc_photogrammetry.reconstruction import plane_endpoint_provenance
from mc_photogrammetry.reconstruction import plane_placement as placement

__all__ = ["IMAGE_DERIVED_ENDPOINT_PROVENANCE_SCHEMA",
           "IMAGE_DERIVED_ENDPOINT_PROVENANCE_PRODUCER",
           "ImageDerivedEndpointProvenanceResult",
           "derive_image_derived_endpoint_provenance"]

IMAGE_DERIVED_ENDPOINT_PROVENANCE_SCHEMA = \
    "mc-photogrammetry-image-derived-endpoint-provenance"
IMAGE_DERIVED_ENDPOINT_PROVENANCE_PRODUCER = \
    "m9-o2-image-derived-endpoint-provenance-1"
ENDPOINT_PROVENANCE_TERMINAL = "geometry_unavailable"
_ERRORS = (AttributeError, TypeError, ValueError, IndexError, KeyError,
           OverflowError, ZeroDivisionError, FloatingPointError)


def _sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _hex(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _payload(value: "ImageDerivedEndpointProvenanceResult") -> dict:
    return {
        "schema": value.schema, "producer": value.producer,
        "terminal": value.terminal, "status": value.status,
        "reason": value.reason, "detail": value.detail,
        "classification": value.classification,
        "relative_translations_result_sha256":
            value.relative_translations_result_sha256,
        "vector_domains_result_sha256": value.vector_domains_result_sha256,
        "graph_sha256": value.graph_sha256,
        "relative_policy_sha256": value.relative_policy_sha256,
        "endpoint_policy_sha256": value.endpoint_policy_sha256,
        "frame_ids": list(value.frame_ids),
        "v2a_input_sha256": value.v2a_input_sha256,
        "v2a_result_sha256": value.v2a_result_sha256,
    }


def _deep_replay(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{
            field.name: _deep_replay(getattr(value, field.name))
            for field in dataclasses.fields(value)})
    if isinstance(value, tuple):
        return tuple(_deep_replay(item) for item in value)
    if isinstance(value, list):
        return [_deep_replay(item) for item in value]
    if isinstance(value, dict):
        return {_deep_replay(key): _deep_replay(item)
                for key, item in value.items()}
    return value


def _replay_v2a(value: object) -> plane_endpoint_provenance.V2AEndpointResult:
    if not isinstance(value, plane_endpoint_provenance.V2AEndpointResult):
        raise ValueError("V2-A result has an invalid type")
    replayed = _deep_replay(value)
    if (not isinstance(replayed, plane_endpoint_provenance.V2AEndpointResult)
            or replayed.to_json_bytes() != value.to_json_bytes()):
        raise ValueError("V2-A result does not replay exactly")
    return replayed


@dataclass(frozen=True, slots=True)
class ImageDerivedEndpointProvenanceResult:
    schema: str
    producer: str
    terminal: str
    status: str
    reason: str
    detail: str
    classification: str | None
    relative_translations_result_sha256: str
    vector_domains_result_sha256: str
    graph_sha256: str
    relative_policy_sha256: str
    endpoint_policy_sha256: str
    frame_ids: tuple[str, ...]
    relative_translations_result: relative_mod.ImageDerivedRelativeTranslationsResult
    endpoint_policy: policy_mod.ImageDerivedEndpointProvenancePolicy
    endpoint_provenance_result: plane_endpoint_provenance.V2AEndpointResult | None
    v2a_input_sha256: str
    v2a_result_sha256: str
    result_sha256: str = ""

    def __post_init__(self) -> None:
        if (self.schema != IMAGE_DERIVED_ENDPOINT_PROVENANCE_SCHEMA
                or self.producer != IMAGE_DERIVED_ENDPOINT_PROVENANCE_PRODUCER
                or self.terminal != ENDPOINT_PROVENANCE_TERMINAL
                or self.status not in (
                    "complete", "refused", "budget_exhausted", "unavailable")
                or type(self.reason) is not str or type(self.detail) is not str
                or not all(_hex(item) for item in (
                    self.relative_translations_result_sha256,
                    self.vector_domains_result_sha256, self.graph_sha256,
                    self.relative_policy_sha256, self.endpoint_policy_sha256))
                or not isinstance(self.frame_ids, tuple) or not self.frame_ids
                or len(set(self.frame_ids)) != len(self.frame_ids)
                or any(type(item) is not str or not item for item in self.frame_ids)
                or not isinstance(self.relative_translations_result,
                                  relative_mod.ImageDerivedRelativeTranslationsResult)
                or not isinstance(self.endpoint_policy,
                                  policy_mod.ImageDerivedEndpointProvenancePolicy)):
            raise ValueError("image-derived endpoint provenance is malformed")
        relative = relative_mod.ImageDerivedRelativeTranslationsResult(**{
            field.name: getattr(self.relative_translations_result, field.name)
            for field in dataclasses.fields(
                relative_mod.ImageDerivedRelativeTranslationsResult)})
        policy = policy_mod.ImageDerivedEndpointProvenancePolicy(**{
            field.name: getattr(self.endpoint_policy, field.name)
            for field in dataclasses.fields(
                policy_mod.ImageDerivedEndpointProvenancePolicy)})
        if (self.relative_translations_result_sha256 != relative.result_sha256
                or self.vector_domains_result_sha256
                    != relative.vector_domains_result_sha256
                or self.graph_sha256 != relative.graph_sha256
                or self.relative_policy_sha256 != relative.relative_policy_sha256
                or self.endpoint_policy_sha256 != policy.policy_sha256
                or policy.to_record()
                    != policy_mod.IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY.to_record()
                or relative.relative_policy.to_record()
                    != relative_policy_mod.IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY.to_record()
                or self.frame_ids != relative.frame_ids):
            raise ValueError("endpoint-provenance sources do not bind")
        if self.status == "unavailable":
            if (not self.reason or self.classification is not None
                    or self.endpoint_provenance_result is not None
                    or self.v2a_input_sha256 or self.v2a_result_sha256):
                raise ValueError("unavailable endpoint provenance must be atomic")
        else:
            result = _replay_v2a(self.endpoint_provenance_result)
            vector = relative.vector_domains_result.vector_domain_result
            closure = relative.relative_translation_result
            if (self.v2a_input_sha256 != result.input_sha256
                    or self.v2a_result_sha256 != result.result_sha256
                    or result.status != self.status
                    or result.classification != self.classification
                    or result.budget.to_record() != policy.budget.to_record()
                    or vector is None or closure is None
                    or result.vector_result_sha256 != vector.result_sha256
                    or result.c2rb_result_sha256 != closure.result_sha256
                    or result.canonical_universe_sha256
                        != vector.canonical_reading_universe_sha256):
                raise ValueError("V2-A result does not bind")
            if self.status == "complete" and (self.reason or self.detail):
                raise ValueError("complete endpoint provenance cannot carry refusal")
            if self.status != "complete" and (
                    not self.reason or self.classification is not None):
                raise ValueError("non-complete endpoint provenance is inconsistent")
        digest = _sha(_payload(self))
        if self.result_sha256 not in ("", digest):
            raise ValueError("image-derived endpoint-provenance digest mismatch")
        object.__setattr__(self, "result_sha256", digest)


def _refusal(reason: str, field: str, error: object = "") -> placement.PlacementRefusal:
    detail = {"field": field}
    if error:
        detail["detail"] = str(error)
    return placement._make_refusal(reason, detail)


def _replay(value: object) -> object:
    if not isinstance(value, relative_mod.ImageDerivedRelativeTranslationsResult):
        return _refusal("invalid_input", "relative_translations_result")
    try:
        return relative_mod.ImageDerivedRelativeTranslationsResult(**{
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(
                relative_mod.ImageDerivedRelativeTranslationsResult)})
    except _ERRORS as error:
        return _refusal("provenance_mismatch", "relative_translations_result",
                        type(error).__name__)


def _make(relative: relative_mod.ImageDerivedRelativeTranslationsResult, *,
          status: str, reason: str = "", detail: str = "",
          result: plane_endpoint_provenance.V2AEndpointResult | None = None,
          ) -> ImageDerivedEndpointProvenanceResult:
    policy = policy_mod.IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY
    return ImageDerivedEndpointProvenanceResult(
        IMAGE_DERIVED_ENDPOINT_PROVENANCE_SCHEMA,
        IMAGE_DERIVED_ENDPOINT_PROVENANCE_PRODUCER,
        ENDPOINT_PROVENANCE_TERMINAL, status, reason, detail,
        None if result is None else result.classification,
        relative.result_sha256, relative.vector_domains_result_sha256,
        relative.graph_sha256, relative.relative_policy_sha256,
        policy.policy_sha256, relative.frame_ids, relative, policy, result,
        "" if result is None else result.input_sha256,
        "" if result is None else result.result_sha256)


def _ready_gate(relative: relative_mod.ImageDerivedRelativeTranslationsResult) -> bool:
    result = relative.relative_translation_result
    vector = relative.vector_domains_result.vector_domain_result
    if (result is None or vector is None or result.status != "complete"
            or result.placement_readiness != "ready" or result.connected is not True
            or len(result.components) != 1 or vector.status != "complete"
            or not vector.domains or vector.refusals
            or vector.represented_work != vector.planned_work):
        return False
    component = result.components[0]
    return (component.outcome == "unique_validated"
            and component.cycle_rank >= 1 and component.exhaustive is True
            and not component.failing_chords
            and all(cycle.matching_provenance_count > 0
                    for cycle in component.cycle_residuals)
            and len(component.translation_classes) == 1)


def derive_image_derived_endpoint_provenance(
    *, relative_translations_result: object,
) -> ImageDerivedEndpointProvenanceResult | placement.PlacementRefusal:
    """Run V2-A once with the exact evidence retained by accepted N2."""
    relative = _replay(relative_translations_result)
    if isinstance(relative, placement.PlacementRefusal):
        return relative
    if relative.status != "complete":
        return _make(relative, status="unavailable",
                     reason="relative_translations_" + relative.status,
                     detail=relative.reason)
    if relative.placement_readiness != "ready":
        return _make(relative, status="unavailable",
                     reason="relative_translations_review_required")
    if not _ready_gate(relative):
        return _refusal("provenance_mismatch", "v2a.ready_gate")
    vector_wrapper = relative.vector_domains_result
    shift = vector_wrapper.shift_domains_result
    phase = vector_wrapper.phase_evidence_result
    try:
        graph = shift.policy_graph_result.graph_result
        readings, _evidence, matches, solution, signatures = \
            axis_mod._bind_universe(graph)
        support = shift.axis_support_result.axis_support
        if (support is None or shift.shift_domain_result is None
                or support.frames != solution.frames
                or support.global_anchor != solution.frames[0]):
            raise ValueError("retained support differs from graph solution")
        policy = policy_mod.IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY
        relative_policy = relative.relative_policy
        result = plane_endpoint_provenance.analyze_endpoint_provenance(
            relative.relative_translation_result,
            vector_wrapper.vector_domain_result, support,
            shift.shift_domain_result, matches, phase.phase_evidence,
            readings=readings, signatures=signatures,
            comparison_tolerance=relative_policy.comparison_tolerance,
            closure_tolerance=relative_policy.closure_tolerance,
            budget=policy.budget)
        replayed = _replay_v2a(result)
        if (replayed.budget.to_record() != policy.budget.to_record()
                or replayed.c2rb_result_sha256
                    != relative.relative_translation_result.result_sha256
                or replayed.vector_result_sha256
                    != vector_wrapper.vector_domain_result.result_sha256
                or replayed.canonical_universe_sha256
                    != vector_wrapper.vector_domain_result.canonical_reading_universe_sha256):
            raise ValueError("V2-A result provenance differs from request")
    except _ERRORS as error:
        return _refusal("provenance_mismatch", "endpoint_provenance_result",
                        type(error).__name__)
    if result.status == "complete":
        return _make(relative, status="complete", result=result)
    if result.status == "refused":
        detail = ",".join(item.reason for item in result.refusals)
        return _make(relative, status="refused",
                     reason="endpoint_provenance_refused", detail=detail,
                     result=result)
    if result.status == "budget_exhausted":
        return _make(relative, status="budget_exhausted",
                     reason=str(result.budget_failure_reason), result=result)
    return _refusal("provenance_mismatch", "endpoint_provenance_result.status")
