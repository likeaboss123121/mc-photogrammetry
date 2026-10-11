"""Close accepted image-derived vector domains with the C2-R-B solver."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

from mc_photogrammetry.pipeline import image_derived_axis_support as axis_mod
from mc_photogrammetry.pipeline import image_derived_relative_translation_policy as policy_mod
from mc_photogrammetry.pipeline import image_derived_vector_domain_policy as vector_policy_mod
from mc_photogrammetry.pipeline import image_derived_vector_domains as vector_mod
from mc_photogrammetry.reconstruction import plane_placement as placement
from mc_photogrammetry.reconstruction import plane_relative_translation

__all__ = ["IMAGE_DERIVED_RELATIVE_TRANSLATIONS_SCHEMA",
           "IMAGE_DERIVED_RELATIVE_TRANSLATIONS_PRODUCER",
           "ImageDerivedRelativeTranslationsResult",
           "derive_image_derived_relative_translations"]

IMAGE_DERIVED_RELATIVE_TRANSLATIONS_SCHEMA = \
    "mc-photogrammetry-image-derived-relative-translations"
IMAGE_DERIVED_RELATIVE_TRANSLATIONS_PRODUCER = \
    "m9-n2-image-derived-relative-translations-1"
RELATIVE_TRANSLATIONS_TERMINAL = "geometry_unavailable"
_ERRORS = (AttributeError, TypeError, ValueError, IndexError, KeyError,
           OverflowError, ZeroDivisionError, FloatingPointError)


def _sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _hex(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _payload(value: "ImageDerivedRelativeTranslationsResult") -> dict:
    return {
        "schema": value.schema, "producer": value.producer,
        "terminal": value.terminal, "status": value.status,
        "reason": value.reason, "detail": value.detail,
        "placement_readiness": value.placement_readiness,
        "vector_domains_result_sha256": value.vector_domains_result_sha256,
        "graph_sha256": value.graph_sha256,
        "shift_policy_sha256": value.shift_policy_sha256,
        "vector_policy_sha256": value.vector_policy_sha256,
        "relative_policy_sha256": value.relative_policy_sha256,
        "frame_ids": list(value.frame_ids),
        "c2rb_input_sha256": value.c2rb_input_sha256,
        "c2rb_result_sha256": value.c2rb_result_sha256,
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


def _replay_c2rb(value: object) -> plane_relative_translation.C2RBResult:
    if not isinstance(value, plane_relative_translation.C2RBResult):
        raise ValueError("C2-R-B result has an invalid type")
    replayed = _deep_replay(value)
    if (not isinstance(replayed, plane_relative_translation.C2RBResult)
            or replayed.to_json_bytes() != value.to_json_bytes()):
        raise ValueError("C2-R-B result does not replay exactly")
    return replayed


@dataclass(frozen=True, slots=True)
class ImageDerivedRelativeTranslationsResult:
    schema: str
    producer: str
    terminal: str
    status: str
    reason: str
    detail: str
    placement_readiness: str
    vector_domains_result_sha256: str
    graph_sha256: str
    shift_policy_sha256: str
    vector_policy_sha256: str
    relative_policy_sha256: str
    frame_ids: tuple[str, ...]
    vector_domains_result: vector_mod.ImageDerivedVectorDomainsResult
    relative_policy: policy_mod.ImageDerivedRelativeTranslationPolicy
    relative_translation_result: plane_relative_translation.C2RBResult | None
    c2rb_input_sha256: str
    c2rb_result_sha256: str
    result_sha256: str = ""

    def __post_init__(self) -> None:
        if (self.schema != IMAGE_DERIVED_RELATIVE_TRANSLATIONS_SCHEMA
                or self.producer != IMAGE_DERIVED_RELATIVE_TRANSLATIONS_PRODUCER
                or self.terminal != RELATIVE_TRANSLATIONS_TERMINAL
                or self.status not in (
                    "complete", "refused", "budget_exhausted", "unavailable")
                or self.placement_readiness not in (
                    "ready", "review_required", "unavailable")
                or type(self.reason) is not str or type(self.detail) is not str
                or not all(_hex(item) for item in (
                    self.vector_domains_result_sha256, self.graph_sha256,
                    self.shift_policy_sha256, self.vector_policy_sha256,
                    self.relative_policy_sha256))
                or not isinstance(self.frame_ids, tuple) or not self.frame_ids
                or len(set(self.frame_ids)) != len(self.frame_ids)
                or any(type(item) is not str or not item for item in self.frame_ids)
                or not isinstance(self.vector_domains_result,
                                  vector_mod.ImageDerivedVectorDomainsResult)
                or not isinstance(self.relative_policy,
                                  policy_mod.ImageDerivedRelativeTranslationPolicy)):
            raise ValueError("image-derived relative translations are malformed")
        vector = vector_mod.ImageDerivedVectorDomainsResult(**{
            field.name: getattr(self.vector_domains_result, field.name)
            for field in dataclasses.fields(
                vector_mod.ImageDerivedVectorDomainsResult)})
        policy = policy_mod.ImageDerivedRelativeTranslationPolicy(**{
            field.name: getattr(self.relative_policy, field.name)
            for field in dataclasses.fields(
                policy_mod.ImageDerivedRelativeTranslationPolicy)})
        if (self.vector_domains_result_sha256 != vector.result_sha256
                or self.graph_sha256 != vector.graph_sha256
                or self.shift_policy_sha256 != vector.shift_policy_sha256
                or self.vector_policy_sha256 != vector.vector_policy_sha256
                or self.relative_policy_sha256 != policy.policy_sha256
                or policy.to_record()
                    != policy_mod.IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY.to_record()
                or self.frame_ids != vector.frame_ids
                or policy.comparison_tolerance
                    != vector_policy_mod.IMAGE_DERIVED_VECTOR_DOMAIN_POLICY.comparison_tolerance):
            raise ValueError("relative-translation sources do not bind")
        if self.status == "unavailable":
            if (not self.reason or self.placement_readiness != "unavailable"
                    or self.relative_translation_result is not None
                    or self.c2rb_input_sha256 or self.c2rb_result_sha256):
                raise ValueError("unavailable relative translations must be atomic")
        else:
            result = _replay_c2rb(self.relative_translation_result)
            if (self.c2rb_input_sha256 != result.input_sha256
                    or self.c2rb_result_sha256 != result.result_sha256
                    or result.status != self.status
                    or result.placement_readiness != self.placement_readiness
                    or result.budget.to_record() != policy.budget.to_record()
                    or result.comparison_tolerance != policy.comparison_tolerance
                    or result.closure_tolerance != policy.closure_tolerance
                    or vector.vector_domain_result is None
                    or result.vector_result_sha256
                        != vector.vector_domain_result.result_sha256
                    or result.vector_input_sha256
                        != vector.vector_domain_result.input_sha256):
                raise ValueError("C2-R-B result does not bind")
            if self.status == "complete" and (self.reason or self.detail):
                raise ValueError("complete relative translations cannot carry refusal")
            if self.status != "complete" and not self.reason:
                raise ValueError("non-complete relative translations need a reason")
        digest = _sha(_payload(self))
        if self.result_sha256 not in ("", digest):
            raise ValueError("image-derived relative-translations digest mismatch")
        object.__setattr__(self, "result_sha256", digest)


def _refusal(reason: str, field: str, error: object = "") -> placement.PlacementRefusal:
    detail = {"field": field}
    if error:
        detail["detail"] = str(error)
    return placement._make_refusal(reason, detail)


def _replay(value: object) -> object:
    if not isinstance(value, vector_mod.ImageDerivedVectorDomainsResult):
        return _refusal("invalid_input", "vector_domains_result")
    try:
        return vector_mod.ImageDerivedVectorDomainsResult(**{
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(
                vector_mod.ImageDerivedVectorDomainsResult)})
    except _ERRORS as error:
        return _refusal("provenance_mismatch", "vector_domains_result",
                        type(error).__name__)


def _make(vector: vector_mod.ImageDerivedVectorDomainsResult, *, status: str,
          reason: str = "", detail: str = "", readiness: str = "unavailable",
          result: plane_relative_translation.C2RBResult | None = None,
          ) -> ImageDerivedRelativeTranslationsResult:
    policy = policy_mod.IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY
    return ImageDerivedRelativeTranslationsResult(
        IMAGE_DERIVED_RELATIVE_TRANSLATIONS_SCHEMA,
        IMAGE_DERIVED_RELATIVE_TRANSLATIONS_PRODUCER,
        RELATIVE_TRANSLATIONS_TERMINAL, status, reason, detail, readiness,
        vector.result_sha256, vector.graph_sha256, vector.shift_policy_sha256,
        vector.vector_policy_sha256, policy.policy_sha256, vector.frame_ids,
        vector, policy, result,
        "" if result is None else result.input_sha256,
        "" if result is None else result.result_sha256)


def derive_image_derived_relative_translations(
    *, vector_domains_result: object,
) -> ImageDerivedRelativeTranslationsResult | placement.PlacementRefusal:
    """Run C2-R-B once with the exact evidence retained by accepted M2."""
    vector = _replay(vector_domains_result)
    if isinstance(vector, placement.PlacementRefusal):
        return vector
    if vector.status != "complete":
        return _make(vector, status="unavailable",
                     reason="vector_domains_" + vector.status,
                     detail=vector.reason)
    if vector.vector_domain_result is None:
        return _refusal("provenance_mismatch", "vector_domain_result")
    shift = vector.shift_domains_result
    phase = vector.phase_evidence_result
    if (shift.status != "complete" or phase.status != "complete"
            or shift.shift_domain_result is None
            or shift.axis_support_result.status != "assembled"
            or shift.axis_support_result.axis_support is None):
        return _refusal("provenance_mismatch", "upstream.ready_state")
    try:
        graph = shift.policy_graph_result.graph_result
        readings, _evidence, matches, solution, signatures = \
            axis_mod._bind_universe(graph)
        support = shift.axis_support_result.axis_support
        if (support.frames != solution.frames
                or support.global_anchor != solution.frames[0]):
            raise ValueError("axis support differs from graph solution")
        policy = policy_mod.IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY
        result = plane_relative_translation.solve_relative_translation_graph(
            vector.vector_domain_result, support, shift.shift_domain_result,
            matches, phase.phase_evidence, readings=readings,
            signatures=signatures,
            comparison_tolerance=policy.comparison_tolerance,
            closure_tolerance=policy.closure_tolerance,
            budget=policy.budget)
        replayed = _replay_c2rb(result)
        if (replayed.vector_result_sha256
                != vector.vector_domain_result.result_sha256
                or replayed.vector_input_sha256
                    != vector.vector_domain_result.input_sha256
                or replayed.budget.to_record() != policy.budget.to_record()
                or replayed.comparison_tolerance != policy.comparison_tolerance
                or replayed.closure_tolerance != policy.closure_tolerance):
            raise ValueError("C2-R-B result provenance differs from request")
    except _ERRORS as error:
        return _refusal("provenance_mismatch", "relative_translation_result",
                        type(error).__name__)
    if result.status == "complete":
        return _make(vector, status="complete",
                     readiness=result.placement_readiness, result=result)
    if result.status == "refused":
        detail = ",".join(item.reason for item in result.refusals)
        return _make(vector, status="refused",
                     reason="relative_translation_refused", detail=detail,
                     result=result)
    if result.status == "budget_exhausted":
        return _make(vector, status="budget_exhausted",
                     reason=str(result.budget_failure_reason), result=result)
    return _refusal("provenance_mismatch", "relative_translation_result.status")
