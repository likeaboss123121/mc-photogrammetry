"""Produce image-derived C2-R vector domains from accepted K and L records."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

from mc_photogrammetry.pipeline import image_derived_axis_support as axis_mod
from mc_photogrammetry.pipeline import image_derived_phase_evidence as phase_mod
from mc_photogrammetry.pipeline import image_derived_shift_domain_policy as shift_policy_mod
from mc_photogrammetry.pipeline import image_derived_shift_domains as shift_mod
from mc_photogrammetry.pipeline import image_derived_vector_domain_policy as policy_mod
from mc_photogrammetry.reconstruction import plane_placement as placement
from mc_photogrammetry.reconstruction import plane_vector_domain

__all__ = ["IMAGE_DERIVED_VECTOR_DOMAINS_SCHEMA",
           "IMAGE_DERIVED_VECTOR_DOMAINS_PRODUCER",
           "ImageDerivedVectorDomainsResult",
           "derive_image_derived_vector_domains"]

IMAGE_DERIVED_VECTOR_DOMAINS_SCHEMA = \
    "mc-photogrammetry-image-derived-vector-domains"
IMAGE_DERIVED_VECTOR_DOMAINS_PRODUCER = \
    "m9-m2-image-derived-vector-domains-1"
VECTOR_DOMAINS_TERMINAL = "geometry_unavailable"
_ERRORS = (AttributeError, TypeError, ValueError, IndexError, KeyError,
           OverflowError, ZeroDivisionError, FloatingPointError)


def _sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _hex(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _payload(value: "ImageDerivedVectorDomainsResult") -> dict:
    return {
        "schema": value.schema, "producer": value.producer,
        "terminal": value.terminal, "status": value.status,
        "reason": value.reason, "detail": value.detail,
        "shift_domains_result_sha256": value.shift_domains_result_sha256,
        "phase_evidence_result_sha256": value.phase_evidence_result_sha256,
        "axis_support_result_sha256": value.axis_support_result_sha256,
        "policy_graph_result_sha256": value.policy_graph_result_sha256,
        "graph_sha256": value.graph_sha256,
        "shift_policy_sha256": value.shift_policy_sha256,
        "vector_policy_sha256": value.vector_policy_sha256,
        "max_shift_cells": value.max_shift_cells,
        "frame_ids": list(value.frame_ids),
        "c2r_input_sha256": value.c2r_input_sha256,
        "c2r_result_sha256": value.c2r_result_sha256,
    }


def _replay_c2(value: object) -> plane_vector_domain.C2RVectorDomainResult:
    if not isinstance(value, plane_vector_domain.C2RVectorDomainResult):
        raise ValueError("C2-R result has an invalid type")
    replayed = plane_vector_domain.C2RVectorDomainResult(**{
        field.name: getattr(value, field.name)
        for field in dataclasses.fields(plane_vector_domain.C2RVectorDomainResult)})
    if replayed.to_json_bytes() != value.to_json_bytes():
        raise ValueError("C2-R result does not replay exactly")
    return replayed


@dataclass(frozen=True, slots=True)
class ImageDerivedVectorDomainsResult:
    schema: str
    producer: str
    terminal: str
    status: str
    reason: str
    detail: str
    shift_domains_result_sha256: str
    phase_evidence_result_sha256: str
    axis_support_result_sha256: str
    policy_graph_result_sha256: str
    graph_sha256: str
    shift_policy_sha256: str
    vector_policy_sha256: str
    max_shift_cells: int
    frame_ids: tuple[str, ...]
    shift_domains_result: shift_mod.ImageDerivedShiftDomainsResult
    phase_evidence_result: phase_mod.ImageDerivedPhaseEvidenceResult
    vector_policy: policy_mod.ImageDerivedVectorDomainPolicy
    vector_domain_result: plane_vector_domain.C2RVectorDomainResult | None
    c2r_input_sha256: str
    c2r_result_sha256: str
    result_sha256: str = ""

    def __post_init__(self) -> None:
        if (self.schema != IMAGE_DERIVED_VECTOR_DOMAINS_SCHEMA
                or self.producer != IMAGE_DERIVED_VECTOR_DOMAINS_PRODUCER
                or self.terminal != VECTOR_DOMAINS_TERMINAL
                or self.status not in (
                    "complete", "refused", "budget_exhausted", "unavailable")
                or type(self.reason) is not str or type(self.detail) is not str
                or not all(_hex(value) for value in (
                    self.shift_domains_result_sha256,
                    self.phase_evidence_result_sha256,
                    self.axis_support_result_sha256,
                    self.policy_graph_result_sha256, self.graph_sha256,
                    self.shift_policy_sha256, self.vector_policy_sha256))
                or type(self.max_shift_cells) is not int
                or self.max_shift_cells <= 0
                or not isinstance(self.frame_ids, tuple) or not self.frame_ids
                or len(set(self.frame_ids)) != len(self.frame_ids)
                or any(type(item) is not str or not item for item in self.frame_ids)
                or not isinstance(self.shift_domains_result,
                                  shift_mod.ImageDerivedShiftDomainsResult)
                or not isinstance(self.phase_evidence_result,
                                  phase_mod.ImageDerivedPhaseEvidenceResult)
                or not isinstance(self.vector_policy,
                                  policy_mod.ImageDerivedVectorDomainPolicy)):
            raise ValueError("image-derived vector-domains result is malformed")
        shift = shift_mod.ImageDerivedShiftDomainsResult(**{
            field.name: getattr(self.shift_domains_result, field.name)
            for field in dataclasses.fields(
                shift_mod.ImageDerivedShiftDomainsResult)})
        phase = phase_mod.ImageDerivedPhaseEvidenceResult(**{
            field.name: getattr(self.phase_evidence_result, field.name)
            for field in dataclasses.fields(
                phase_mod.ImageDerivedPhaseEvidenceResult)})
        policy = policy_mod.ImageDerivedVectorDomainPolicy(**{
            field.name: getattr(self.vector_policy, field.name)
            for field in dataclasses.fields(
                policy_mod.ImageDerivedVectorDomainPolicy)})
        if (self.shift_domains_result_sha256 != shift.result_sha256
                or self.phase_evidence_result_sha256 != phase.result_sha256
                or self.axis_support_result_sha256
                    != shift.axis_support_result_sha256
                or self.policy_graph_result_sha256
                    != shift.policy_graph_result_sha256
                or self.policy_graph_result_sha256
                    != phase.policy_graph_result_sha256
                or self.graph_sha256 != shift.graph_sha256
                or self.graph_sha256 != phase.graph_sha256
                or self.shift_policy_sha256 != shift.policy_sha256
                or self.shift_policy_sha256 != phase.policy_sha256
                or self.vector_policy_sha256 != policy.policy_sha256
                or policy.to_record()
                    != policy_mod.IMAGE_DERIVED_VECTOR_DOMAIN_POLICY.to_record()
                or self.max_shift_cells != shift.max_shift_cells
                or self.max_shift_cells != phase.max_shift_cells
                or self.frame_ids != shift.frame_ids
                or self.frame_ids != phase.frame_ids):
            raise ValueError("vector-domain sources do not bind")
        if self.status == "unavailable":
            if (not self.reason or self.vector_domain_result is not None
                    or self.c2r_input_sha256 or self.c2r_result_sha256):
                raise ValueError("unavailable vector domains must be atomic")
        else:
            c2r = _replay_c2(self.vector_domain_result)
            if (self.c2r_input_sha256 != c2r.input_sha256
                    or self.c2r_result_sha256 != c2r.result_sha256
                    or c2r.status != self.status
                    or c2r.budget.to_record() != policy.budget.to_record()):
                raise ValueError("C2-R result does not bind")
            if self.status == "complete" and (self.reason or self.detail):
                raise ValueError("complete vector domains cannot carry a refusal")
            if self.status != "complete" and not self.reason:
                raise ValueError("non-complete vector domains need a reason")
        digest = _sha(_payload(self))
        if self.result_sha256 not in ("", digest):
            raise ValueError("image-derived vector-domains digest mismatch")
        object.__setattr__(self, "result_sha256", digest)


def _refusal(reason: str, field: str, error: object = "") -> placement.PlacementRefusal:
    detail = {"field": field}
    if error:
        detail["detail"] = str(error)
    return placement._make_refusal(reason, detail)


def _replay(value: object, expected: type, field: str) -> object:
    if not isinstance(value, expected):
        return _refusal("invalid_input", field)
    try:
        return expected(**{item.name: getattr(value, item.name)
                           for item in dataclasses.fields(expected)})
    except _ERRORS as error:
        return _refusal("provenance_mismatch", field, type(error).__name__)


def _make(shift: shift_mod.ImageDerivedShiftDomainsResult,
          phase: phase_mod.ImageDerivedPhaseEvidenceResult, *, status: str,
          reason: str = "", detail: str = "",
          c2r: plane_vector_domain.C2RVectorDomainResult | None = None,
          ) -> ImageDerivedVectorDomainsResult:
    policy = policy_mod.IMAGE_DERIVED_VECTOR_DOMAIN_POLICY
    return ImageDerivedVectorDomainsResult(
        IMAGE_DERIVED_VECTOR_DOMAINS_SCHEMA,
        IMAGE_DERIVED_VECTOR_DOMAINS_PRODUCER, VECTOR_DOMAINS_TERMINAL,
        status, reason, detail, shift.result_sha256, phase.result_sha256,
        shift.axis_support_result_sha256, shift.policy_graph_result_sha256,
        shift.graph_sha256, shift.policy_sha256, policy.policy_sha256,
        shift.max_shift_cells, shift.frame_ids, shift, phase, policy, c2r,
        "" if c2r is None else c2r.input_sha256,
        "" if c2r is None else c2r.result_sha256)


def derive_image_derived_vector_domains(
    *, shift_domains_result: object, phase_evidence_result: object,
) -> ImageDerivedVectorDomainsResult | placement.PlacementRefusal:
    """Run accepted C2-R once with the exact K/L evidence and frozen policy."""
    shift = _replay(shift_domains_result,
                    shift_mod.ImageDerivedShiftDomainsResult,
                    "shift_domains_result")
    if isinstance(shift, placement.PlacementRefusal):
        return shift
    phase = _replay(phase_evidence_result,
                    phase_mod.ImageDerivedPhaseEvidenceResult,
                    "phase_evidence_result")
    if isinstance(phase, placement.PlacementRefusal):
        return phase
    if (shift.policy_graph_result_sha256
            != phase.policy_graph_result_sha256
            or shift.graph_sha256 != phase.graph_sha256
            or shift.policy_sha256 != phase.policy_sha256
            or shift.max_shift_cells != phase.max_shift_cells
            or shift.frame_ids != phase.frame_ids
            or shift.policy_graph_result.graph_result.session_sha256
                != shift.axis_support_result.session_sha256
            or shift.policy_graph_result.graph_result.lattice_sha256
                != shift.axis_support_result.lattice_sha256
            or shift.policy_graph_result.graph_result.signatures_sha256
                != shift.axis_support_result.signatures_sha256
            or shift.policy_graph_result.policy.to_record()
                != shift_policy_mod.IMAGE_DERIVED_SHIFT_DOMAIN_POLICY.to_record()):
        return _refusal("provenance_mismatch", "upstream.binding")
    if shift.status != "complete":
        return _make(shift, phase, status="unavailable",
                     reason="shift_domains_" + shift.status,
                     detail=shift.reason)
    if phase.status != "complete":
        return _make(shift, phase, status="unavailable",
                     reason="phase_evidence_" + phase.status,
                     detail=phase.reason)
    if (shift.axis_support_result.status != "assembled"
            or shift.axis_support_result.axis_support is None
            or shift.shift_domain_result is None):
        return _refusal("provenance_mismatch", "upstream.ready_state")
    graph = shift.policy_graph_result.graph_result
    try:
        readings, _evidence, matches, solution, signatures = \
            axis_mod._bind_universe(graph)
        support = shift.axis_support_result.axis_support
        if (support.frames != solution.frames
                or support.global_anchor != solution.frames[0]):
            raise ValueError("axis support differs from graph solution")
        policy = policy_mod.IMAGE_DERIVED_VECTOR_DOMAIN_POLICY
        result = plane_vector_domain.build_lateral_vector_domains(
            support, shift.shift_domain_result, matches, phase.phase_evidence,
            readings=readings, signatures=signatures,
            comparison_tolerance=policy.comparison_tolerance,
            budget=policy.budget)
        replayed = _replay_c2(result)
        if (replayed.budget.to_record() != policy.budget.to_record()
                or not plane_vector_domain.validate_lateral_vector_domain_inputs(
                    replayed, support, shift.shift_domain_result, matches,
                    phase.phase_evidence, readings=readings,
                    signatures=signatures,
                    comparison_tolerance=policy.comparison_tolerance)):
            raise ValueError("C2-R result input validation failed")
    except _ERRORS as error:
        return _refusal("provenance_mismatch", "vector_domain_result",
                        type(error).__name__)
    if result.status == "complete":
        return _make(shift, phase, status="complete", c2r=result)
    if result.status == "refused":
        detail = ",".join(item.reason for item in result.refusals)
        return _make(shift, phase, status="refused",
                     reason="vector_domain_refused", detail=detail, c2r=result)
    if result.status == "budget_exhausted":
        return _make(shift, phase, status="budget_exhausted",
                     reason=str(result.budget_failure_reason), c2r=result)
    return _refusal("provenance_mismatch", "vector_domain_result.status")
