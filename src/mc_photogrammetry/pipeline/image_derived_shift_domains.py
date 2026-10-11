"""Produce policy-bound P1-D shift domains from accepted H and J records."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

from mc_photogrammetry.pipeline import image_derived_axis_support as axis_mod
from mc_photogrammetry.pipeline import image_derived_graph_policy_v5 as graph_policy_mod
from mc_photogrammetry.pipeline import image_derived_shift_domain_policy as policy_mod
from mc_photogrammetry.reconstruction import plane_placement as placement
from mc_photogrammetry.reconstruction import plane_graph, plane_shift_domain

__all__ = ["IMAGE_DERIVED_SHIFT_DOMAINS_SCHEMA",
           "IMAGE_DERIVED_SHIFT_DOMAINS_PRODUCER",
           "ImageDerivedShiftDomainsResult", "derive_image_derived_shift_domains"]

IMAGE_DERIVED_SHIFT_DOMAINS_SCHEMA = "mc-photogrammetry-image-derived-shift-domains-v2"
IMAGE_DERIVED_SHIFT_DOMAINS_PRODUCER = "m9-k-image-derived-shift-domains-2"
SHIFT_DOMAINS_TERMINAL = "geometry_unavailable"
_ERRORS = (AttributeError, TypeError, ValueError, IndexError, KeyError,
           OverflowError, ZeroDivisionError, FloatingPointError)


def _sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hex(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _p1d_sha(value: plane_shift_domain.ShiftDomainResult | None) -> str:
    return "" if value is None else _sha(value.to_record())


def _payload(value: "ImageDerivedShiftDomainsResult") -> dict:
    return {
        "schema": value.schema, "producer": value.producer,
        "terminal": value.terminal, "status": value.status,
        "reason": value.reason, "detail": value.detail,
        "axis_support_result_sha256": value.axis_support_result_sha256,
        "policy_graph_result_sha256": value.policy_graph_result_sha256,
        "graph_sha256": value.graph_sha256,
        "policy_sha256": value.policy_sha256,
        "max_shift_cells": value.max_shift_cells,
        "correlation_algorithm_id": value.correlation_algorithm_id,
        "frame_ids": list(value.frame_ids),
        "p1d_input_sha256": value.p1d_input_sha256,
        "p1d_result_sha256": value.p1d_result_sha256,
    }


@dataclass(frozen=True, slots=True)
class ImageDerivedShiftDomainsResult:
    schema: str
    producer: str
    terminal: str
    status: str
    reason: str
    detail: str
    axis_support_result_sha256: str
    policy_graph_result_sha256: str
    graph_sha256: str
    policy_sha256: str
    max_shift_cells: int
    correlation_algorithm_id: str
    frame_ids: tuple[str, ...]
    axis_support_result: axis_mod.ImageDerivedAxisSupportResult
    policy_graph_result: graph_policy_mod.ImageDerivedPolicyBoundGraph
    shift_domain_result: plane_shift_domain.ShiftDomainResult | None
    p1d_input_sha256: str
    p1d_result_sha256: str
    result_sha256: str = ""

    def __post_init__(self) -> None:
        if (self.schema != IMAGE_DERIVED_SHIFT_DOMAINS_SCHEMA
                or self.producer != IMAGE_DERIVED_SHIFT_DOMAINS_PRODUCER
                or self.terminal != SHIFT_DOMAINS_TERMINAL
                or self.status not in (
                    "complete", "refused", "budget_exhausted", "unavailable")
                or type(self.reason) is not str or type(self.detail) is not str
                or not all(_hex(item) for item in (
                    self.axis_support_result_sha256,
                    self.policy_graph_result_sha256, self.graph_sha256,
                    self.policy_sha256))
                or type(self.max_shift_cells) is not int
                or self.max_shift_cells <= 0
                or self.correlation_algorithm_id
                != plane_graph.PLANE_CORRELATION_ALGORITHM
                or not isinstance(self.frame_ids, tuple) or not self.frame_ids
                or len(set(self.frame_ids)) != len(self.frame_ids)
                or any(type(item) is not str or not item for item in self.frame_ids)
                or not isinstance(self.axis_support_result,
                                  axis_mod.ImageDerivedAxisSupportResult)
                or not isinstance(self.policy_graph_result,
                                  graph_policy_mod.ImageDerivedPolicyBoundGraph)):
            raise ValueError("image-derived shift-domains result is malformed")
        axis_replay = axis_mod.ImageDerivedAxisSupportResult(**{
            field.name: getattr(self.axis_support_result, field.name)
            for field in dataclasses.fields(axis_mod.ImageDerivedAxisSupportResult)})
        graph_replay = graph_policy_mod.ImageDerivedPolicyBoundGraph(**{
            field.name: getattr(self.policy_graph_result, field.name)
            for field in dataclasses.fields(graph_policy_mod.ImageDerivedPolicyBoundGraph)})
        if self.status != "unavailable":
            graph_policy_mod._require_live_policy(self.policy_graph_result)
        if (self.axis_support_result_sha256 != axis_replay.result_sha256
                or self.policy_graph_result_sha256 != graph_replay.result_sha256
                or self.graph_sha256 != axis_replay.graph_sha256
                or self.graph_sha256 != graph_replay.graph_sha256
                or self.policy_sha256 != graph_replay.policy_sha256
                or self.max_shift_cells != graph_replay.max_shift_cells
                or self.max_shift_cells != axis_replay.max_shift_cells
                or self.correlation_algorithm_id
                != graph_replay.correlation_algorithm
                or self.correlation_algorithm_id
                != graph_replay.policy.correlation_algorithm_id
                or self.correlation_algorithm_id
                != axis_replay.correlation_algorithm_id
                or self.frame_ids != axis_replay.frame_ids
                or self.frame_ids != graph_replay.frame_ids):
            raise ValueError("shift-domain sources do not bind")
        if self.status == "unavailable":
            if (not self.reason or self.shift_domain_result is not None
                    or self.p1d_input_sha256 or self.p1d_result_sha256):
                raise ValueError("unavailable shift domains must be atomic")
        else:
            if not isinstance(self.shift_domain_result,
                              plane_shift_domain.ShiftDomainResult):
                raise ValueError("shift domains need a typed P1-D result")
            replayed = plane_shift_domain.ShiftDomainResult(**{
                field.name: getattr(self.shift_domain_result, field.name)
                for field in dataclasses.fields(plane_shift_domain.ShiftDomainResult)})
            if (replayed.to_record() != self.shift_domain_result.to_record()
                    or replayed.schema != plane_shift_domain.SHIFT_DOMAIN_SCHEMA
                    or replayed.version != plane_shift_domain.SHIFT_DOMAIN_VERSION
                    or replayed.correlation_algorithm_id
                    != self.correlation_algorithm_id
                    or self.p1d_input_sha256 != replayed.input_sha256
                    or self.p1d_result_sha256 != _p1d_sha(replayed)
                    or replayed.status != self.status):
                raise ValueError("P1-D result does not replay")
            if self.status == "complete" and (self.reason or self.detail):
                raise ValueError("complete shift domains cannot carry a refusal")
            if self.status != "complete" and not self.reason:
                raise ValueError("non-complete shift domains need a reason")
        digest = _sha(_payload(self))
        if self.result_sha256 not in ("", digest):
            raise ValueError("image-derived shift-domains digest mismatch")
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
        expected(**{item.name: getattr(value, item.name)
                    for item in dataclasses.fields(expected)})
        if expected is graph_policy_mod.ImageDerivedPolicyBoundGraph:
            graph_policy_mod._require_live_policy(value)
    except _ERRORS as error:
        return _refusal("provenance_mismatch", field, type(error).__name__)
    return value


def _make(axis: axis_mod.ImageDerivedAxisSupportResult,
          graph: graph_policy_mod.ImageDerivedPolicyBoundGraph, *,
          status: str, reason: str = "", detail: str = "",
          p1d: plane_shift_domain.ShiftDomainResult | None = None,
          ) -> ImageDerivedShiftDomainsResult:
    return ImageDerivedShiftDomainsResult(
        IMAGE_DERIVED_SHIFT_DOMAINS_SCHEMA, IMAGE_DERIVED_SHIFT_DOMAINS_PRODUCER,
        SHIFT_DOMAINS_TERMINAL, status, reason, detail, axis.result_sha256,
        graph.result_sha256, graph.graph_sha256, graph.policy_sha256,
        graph.max_shift_cells, graph.correlation_algorithm, graph.frame_ids,
        axis, graph, p1d,
        "" if p1d is None else p1d.input_sha256, _p1d_sha(p1d))


def derive_image_derived_shift_domains(
    *, axis_support_result: object, policy_graph_result: object,
) -> ImageDerivedShiftDomainsResult | placement.PlacementRefusal:
    """Run accepted P1-D once with the exact H/J evidence and frozen policy."""
    axis = _replay(axis_support_result, axis_mod.ImageDerivedAxisSupportResult,
                   "axis_support_result")
    if isinstance(axis, placement.PlacementRefusal):
        return axis
    graph = _replay(policy_graph_result,
                    graph_policy_mod.ImageDerivedPolicyBoundGraph,
                    "policy_graph_result")
    if isinstance(graph, placement.PlacementRefusal):
        return graph
    if (axis.graph_sha256 != graph.graph_sha256
            or axis.frame_ids != graph.frame_ids
            or axis.session_sha256 != graph.graph_result.session_sha256
            or axis.lattice_sha256 != graph.graph_result.lattice_sha256
            or axis.signatures_sha256 != graph.graph_result.signatures_sha256
            or axis.correlation_algorithm_id != graph.correlation_algorithm
            or axis.max_shift_cells != graph.max_shift_cells
            or graph.policy.correlation_algorithm_id
            != graph.correlation_algorithm
            or graph.policy.to_record()
            != policy_mod.IMAGE_DERIVED_SHIFT_DOMAIN_POLICY.to_record()):
        return _refusal("provenance_mismatch", "upstream.binding")
    if axis.status != "assembled":
        return _make(axis, graph, status="unavailable",
                     reason="axis_support_unavailable", detail=axis.reason)
    if graph.status != "bound":
        return _make(axis, graph, status="unavailable",
                     reason="policy_graph_unavailable", detail=graph.reason)
    if axis.axis_support is None:
        return _refusal("provenance_mismatch", "axis_support_result.axis_support")
    try:
        readings, _evidence, matches, solution, raw = \
            axis_mod._bind_universe(graph.graph_result)
    except _ERRORS as error:
        return _refusal("provenance_mismatch", "graph_result.binding",
                        type(error).__name__)
    if (axis.axis_support.frames != solution.frames
            or axis.axis_support.global_anchor != solution.frames[0]):
        return _refusal("provenance_mismatch", "axis_support_result.binding")
    try:
        result = plane_shift_domain.build_shift_domains(
            axis.axis_support, matches, raw, readings=readings,
            max_shift_cells=graph.max_shift_cells, budget=graph.policy.budget)
        if not isinstance(result, plane_shift_domain.ShiftDomainResult):
            return _refusal("provenance_mismatch", "shift_domain_result.type")
        replayed = plane_shift_domain.ShiftDomainResult(**{
            field.name: getattr(result, field.name)
            for field in dataclasses.fields(plane_shift_domain.ShiftDomainResult)})
    except _ERRORS as error:
        return _refusal("provenance_mismatch", "shift_domain_result",
                        type(error).__name__)
    if (replayed.to_record() != result.to_record()
            or replayed.budget.to_record() != graph.policy.budget.to_record()):
        return _refusal("provenance_mismatch", "shift_domain_result.replay")
    if (result.status == "complete"
            and not plane_shift_domain.validate_shift_domain_inputs(
                result, axis.axis_support, matches, raw, readings=readings,
                max_shift_cells=graph.max_shift_cells)):
        return _refusal("provenance_mismatch", "shift_domain_result.input")
    if result.status == "complete":
        return _make(axis, graph, status="complete", p1d=result)
    if result.status == "refused":
        detail = ",".join(item.reason for item in result.refusals)
        return _make(axis, graph, status="refused",
                     reason="shift_domain_refused", detail=detail, p1d=result)
    if result.status == "budget_exhausted":
        return _make(axis, graph, status="budget_exhausted",
                     reason=str(result.budget_failure_reason), p1d=result)
    return _refusal("provenance_mismatch", "shift_domain_result.status")
