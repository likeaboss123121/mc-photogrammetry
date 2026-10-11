"""Replay an accepted image-derived graph under the frozen P1-D radius."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

from mc_photogrammetry.pipeline import image_derived_axis_support as axis_adapter
from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
from mc_photogrammetry.pipeline import image_derived_shift_domain_policy as policy_mod
from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction import plane_placement as placement

__all__ = ["HISTORICAL_IMAGE_DERIVED_GRAPH_POLICY_SCHEMA",
           "HISTORICAL_IMAGE_DERIVED_GRAPH_POLICY_PRODUCER",
           "HistoricalImageDerivedPolicyBoundGraph", "bind_image_derived_graph_policy"]

HISTORICAL_IMAGE_DERIVED_GRAPH_POLICY_SCHEMA = "mc-photogrammetry-image-derived-graph-policy-v3"
HISTORICAL_IMAGE_DERIVED_GRAPH_POLICY_PRODUCER = "m9-j-image-derived-graph-policy-3"
GRAPH_POLICY_TERMINAL = "geometry_unavailable"
_ERRORS = (AttributeError, TypeError, ValueError, IndexError, KeyError,
           OverflowError, ZeroDivisionError, FloatingPointError)


def _sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hex(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _payload(value: "HistoricalImageDerivedPolicyBoundGraph") -> dict:
    return {
        "schema": value.schema, "producer": value.producer,
        "terminal": value.terminal, "status": value.status,
        "reason": value.reason, "detail": value.detail,
        "graph_sha256": value.graph_sha256,
        "policy_sha256": value.policy_sha256,
        "max_shift_cells": value.max_shift_cells,
        "correlation_algorithm": value.correlation_algorithm,
        "frame_ids": list(value.frame_ids),
        "graph_result_sha256": value.graph_result.result_sha256,
        "policy": value.policy.to_record(),
    }


@dataclass(frozen=True, slots=True)
class HistoricalImageDerivedPolicyBoundGraph:
    schema: str
    producer: str
    terminal: str
    status: str
    reason: str
    detail: str
    graph_sha256: str
    policy_sha256: str
    max_shift_cells: int
    frame_ids: tuple[str, ...]
    graph_result: graph_mod.HistoricalImageDerivedGraphResult
    policy: policy_mod.ImageDerivedShiftDomainPolicy
    correlation_algorithm: str = plane_graph.PLANE_CORRELATION_ALGORITHM
    result_sha256: str = ""

    def __post_init__(self) -> None:
        if (self.schema != HISTORICAL_IMAGE_DERIVED_GRAPH_POLICY_SCHEMA
                or self.producer != HISTORICAL_IMAGE_DERIVED_GRAPH_POLICY_PRODUCER
                or self.terminal != GRAPH_POLICY_TERMINAL
                or self.status not in ("bound", "unavailable")
                or type(self.reason) is not str or type(self.detail) is not str
                or not _hex(self.graph_sha256) or not _hex(self.policy_sha256)
                or type(self.max_shift_cells) is not int
                or self.max_shift_cells <= 0
                or self.correlation_algorithm != plane_graph.PLANE_CORRELATION_ALGORITHM
                or not isinstance(self.frame_ids, tuple) or not self.frame_ids
                or len(set(self.frame_ids)) != len(self.frame_ids)
                or any(type(item) is not str or not item for item in self.frame_ids)
                or not isinstance(self.graph_result,
                                  graph_mod.HistoricalImageDerivedGraphResult)
                or not isinstance(self.policy,
                                  policy_mod.ImageDerivedShiftDomainPolicy)):
            raise ValueError("image-derived policy-bound graph is malformed")
        graph = graph_mod.HistoricalImageDerivedGraphResult(**{
            field.name: getattr(self.graph_result, field.name)
            for field in dataclasses.fields(graph_mod.HistoricalImageDerivedGraphResult)})
        policy = policy_mod.ImageDerivedShiftDomainPolicy(**{
            field.name: getattr(self.policy, field.name)
            for field in dataclasses.fields(policy_mod.ImageDerivedShiftDomainPolicy)})
        if (self.graph_sha256 != graph.result_sha256
                or self.policy_sha256 != policy.policy_sha256
                or self.max_shift_cells != policy.max_shift_cells
                or graph.max_shift_cells != self.max_shift_cells
                or graph.correlation_algorithm != self.correlation_algorithm
                or policy.correlation_algorithm_id != self.correlation_algorithm
                or self.frame_ids != graph.frame_ids
                or policy.to_record()
                != policy_mod.IMAGE_DERIVED_SHIFT_DOMAIN_POLICY.to_record()):
            raise ValueError("policy-bound graph sources do not replay")
        if self.status == "bound":
            if self.reason or self.detail or graph.status != "assembled":
                raise ValueError("bound graph status is inconsistent")
        elif not self.reason or graph.status == "assembled":
            raise ValueError("unavailable graph status is inconsistent")
        digest = _sha(_payload(self))
        if self.result_sha256 not in ("", digest):
            raise ValueError("policy-bound graph digest mismatch")
        object.__setattr__(self, "result_sha256", digest)


def _refusal(reason: str, field: str, error: object = "") -> placement.PlacementRefusal:
    detail = {"field": field}
    if error:
        detail["detail"] = str(error)
    return placement._make_refusal(reason, detail)


def _replay_graph(value: object) -> graph_mod.HistoricalImageDerivedGraphResult | placement.PlacementRefusal:
    if not isinstance(value, graph_mod.HistoricalImageDerivedGraphResult):
        return _refusal("invalid_input", "graph_result")
    try:
        graph_mod.HistoricalImageDerivedGraphResult(**{
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(graph_mod.HistoricalImageDerivedGraphResult)})
    except _ERRORS as error:
        return _refusal("provenance_mismatch", "graph_result",
                        type(error).__name__)
    return value


def _replay_policy() -> policy_mod.ImageDerivedShiftDomainPolicy | placement.PlacementRefusal:
    source = policy_mod.IMAGE_DERIVED_SHIFT_DOMAIN_POLICY
    try:
        replayed = policy_mod.ImageDerivedShiftDomainPolicy(**{
            field.name: getattr(source, field.name)
            for field in dataclasses.fields(policy_mod.ImageDerivedShiftDomainPolicy)})
    except _ERRORS as error:
        return _refusal("provenance_mismatch", "policy",
                        type(error).__name__)
    if replayed.to_record() != policy_mod._builtin_policy().to_record():
        return _refusal("provenance_mismatch", "policy.builtin")
    if (replayed.correlation_algorithm_id
            != plane_graph.PLANE_CORRELATION_ALGORITHM):
        return _refusal("provenance_mismatch", "policy.correlation_algorithm_id")
    return source


def _result(graph: graph_mod.HistoricalImageDerivedGraphResult,
            policy: policy_mod.ImageDerivedShiftDomainPolicy, *,
            status: str, reason: str = "", detail: str = "",
            ) -> HistoricalImageDerivedPolicyBoundGraph:
    return HistoricalImageDerivedPolicyBoundGraph(
        HISTORICAL_IMAGE_DERIVED_GRAPH_POLICY_SCHEMA, HISTORICAL_IMAGE_DERIVED_GRAPH_POLICY_PRODUCER,
        GRAPH_POLICY_TERMINAL, status, reason, detail, graph.result_sha256,
        policy.policy_sha256, policy.max_shift_cells, graph.frame_ids,
        graph, policy, plane_graph.PLANE_CORRELATION_ALGORITHM)


def bind_historical_image_derived_graph_policy(
    *, graph_result: object,
) -> HistoricalImageDerivedPolicyBoundGraph | placement.PlacementRefusal:
    """Require exact graph-match replay under the sole built-in P1-D policy."""
    graph = _replay_graph(graph_result)
    if isinstance(graph, placement.PlacementRefusal):
        return graph
    policy = _replay_policy()
    if isinstance(policy, placement.PlacementRefusal):
        return policy
    if graph.status != "assembled":
        return _result(graph, policy, status="unavailable",
                       reason="graph_unavailable", detail=graph.reason)
    try:
        readings = graph.readings
        raw = tuple(e.signature for e in graph.signatures)
        replayed = plane_graph.match_plane_graph(
            readings, raw, max_shift_cells=policy.max_shift_cells)
    except _ERRORS as error:
        return _refusal("provenance_mismatch", "graph_result.policy_replay",
                        type(error).__name__)
    if not isinstance(replayed, tuple) or replayed != graph.matches:
        return _refusal(
            "provenance_mismatch", "graph_result.matches.policy_replay")
    return _result(graph, policy, status="bound")


IMAGE_DERIVED_GRAPH_POLICY_SCHEMA = "mc-photogrammetry-image-derived-graph-policy-v5"
IMAGE_DERIVED_GRAPH_POLICY_PRODUCER = "m9-j-image-derived-graph-policy-5"
__all__ += ["IMAGE_DERIVED_GRAPH_POLICY_SCHEMA", "IMAGE_DERIVED_GRAPH_POLICY_PRODUCER",
            "ImageDerivedPolicyBoundGraph", "bind_image_derived_graph_policy",
            "bind_historical_image_derived_graph_policy"]


def __getattr__(name):
    if name == "ImageDerivedPolicyBoundGraph":
        from mc_photogrammetry.pipeline.image_derived_graph_policy_v5 import ImageDerivedPolicyBoundGraph
        return ImageDerivedPolicyBoundGraph
    raise AttributeError(name)


def bind_image_derived_graph_policy(*, graph_result):
    """Production binding requires the accepted producer7 live capability."""
    from mc_photogrammetry.pipeline import image_derived_graph_policy_v5
    return image_derived_graph_policy_v5.bind_image_derived_graph_policy(graph_result=graph_result)
