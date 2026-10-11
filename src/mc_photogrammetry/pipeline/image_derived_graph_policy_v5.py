"""Production policy binding of complete candidates and joint certificate.

Binding a live graph performs no matcher or numerical evaluation. Explicit
source replay reconstructs the complete candidate universe once, rather than
treating the representative geometry constraints as the matcher output.
"""
from __future__ import annotations

from dataclasses import dataclass

from mc_photogrammetry.pipeline import image_derived_graph as legacy
from mc_photogrammetry.pipeline import image_derived_joint_graph as graph_mod
from mc_photogrammetry.pipeline import image_derived_graph_policy_v4 as phase_b
from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp
from mc_photogrammetry.reconstruction import plane_graph, plane_placement as placement

IMAGE_DERIVED_GRAPH_POLICY_SCHEMA = "mc-photogrammetry-image-derived-graph-policy-v5"
IMAGE_DERIVED_GRAPH_POLICY_PRODUCER = "m9-j-image-derived-graph-policy-5"
GRAPH_POLICY_TERMINAL = "geometry_unavailable"
_ERRORS = graph_mod._ERRORS


def _validated_policy(value):
    expected = phase_b._policy()
    if type(value) is not type(expected):
        raise ValueError("foreign production shift policy")
    type(expected)(**legacy._fields(value))
    if value.to_record() != expected.to_record():
        raise ValueError("production policy differs from unchanged built-in")
    return value


def _payload(value):
    return {"schema": value.schema, "producer": value.producer,
            "terminal": value.terminal, "status": value.status, "reason": value.reason, "detail": value.detail,
            "graph_sha256": value.graph_sha256, "selection_sha256": value.graph_result.selection.result_sha256,
            "candidate_matches_sha256": value.graph_result.selection.candidate_matches_sha256,
            "policy_sha256": value.policy_sha256, "max_shift_cells": value.max_shift_cells,
            "correlation_algorithm": value.correlation_algorithm, "frame_ids": value.frame_ids,
            "policy": value.policy.to_record()}


@dataclass(frozen=True, slots=True)
class ImageDerivedPolicyBoundGraph:
    schema: str
    producer: str
    terminal: str
    status: str
    reason: str
    detail: str
    graph_sha256: str
    policy_sha256: str
    max_shift_cells: int
    frame_ids: tuple
    graph_result: graph_mod.ImageDerivedGraphResult
    policy: object
    correlation_algorithm: str = plane_graph.PLANE_CORRELATION_ALGORITHM
    result_sha256: str = ""

    def __post_init__(self):
        if (self.schema != IMAGE_DERIVED_GRAPH_POLICY_SCHEMA or self.producer != IMAGE_DERIVED_GRAPH_POLICY_PRODUCER
                or self.terminal != GRAPH_POLICY_TERMINAL or self.status not in ("bound", "unavailable")
                or type(self.graph_result) is not graph_mod.ImageDerivedGraphResult
                or type(self.reason) is not str or type(self.detail) is not str
                or type(self.max_shift_cells) is not int or self.max_shift_cells != 32
                or self.correlation_algorithm != plane_graph.PLANE_CORRELATION_ALGORITHM):
            raise ValueError("production joint policy identity malformed")
        graph_mod.ImageDerivedGraphResult(**legacy._fields(self.graph_result))
        _validated_policy(self.policy)
        graph = self.graph_result
        if (self.graph_sha256 != graph.result_sha256 or self.policy_sha256 != self.policy.policy_sha256
                or self.max_shift_cells != self.policy.max_shift_cells or self.frame_ids != graph.frame_ids
                or graph.max_shift_cells != self.max_shift_cells):
            raise ValueError("production joint policy sources differ")
        if self.status == "bound":
            if graph.status != "assembled" or self.reason or self.detail:
                raise ValueError("production bound policy is inconsistent")
        elif graph.status == "assembled" or self.reason != "graph_unavailable" or self.detail != graph.reason:
            raise ValueError("unavailable production policy is inconsistent")
        digest = selection._sha(_payload(self))
        if self.result_sha256 not in ("", digest):
            raise ValueError("production joint policy digest mismatch")
        object.__setattr__(self, "result_sha256", digest)


def _require_live_policy(value):
    """Structural records are not an execution capability, including bound ones."""
    if type(value) is not ImageDerivedPolicyBoundGraph:
        raise ValueError("foreign production policy")
    ImageDerivedPolicyBoundGraph(**legacy._fields(value))
    graph_mod._require_live(value.graph_result)
    return value


def bind_image_derived_graph_policy(*, graph_result):
    """Bind the live graph's complete candidate certificate without a second match."""
    if type(graph_result) is not graph_mod.ImageDerivedGraphResult:
        return legacy._refusal("provenance_mismatch", "joint_policy.graph_identity")
    try:
        graph_mod.ImageDerivedGraphResult(**legacy._fields(graph_result))
        graph_mod._require_live(graph_result)
        policy = phase_b.policy_mod.IMAGE_DERIVED_SHIFT_DOMAIN_POLICY
        _validated_policy(policy)
        assembled = graph_result.status == "assembled"
        return ImageDerivedPolicyBoundGraph(IMAGE_DERIVED_GRAPH_POLICY_SCHEMA, IMAGE_DERIVED_GRAPH_POLICY_PRODUCER,
                    GRAPH_POLICY_TERMINAL, "bound" if assembled else "unavailable", "" if assembled else "graph_unavailable",
                    "" if assembled else graph_result.reason, graph_result.result_sha256, policy.policy_sha256,
                    policy.max_shift_cells, graph_result.frame_ids, graph_result, policy)
    except _ERRORS as error:
        return legacy._refusal("provenance_mismatch", "joint_policy.binding", type(error).__name__)


def replay_image_derived_graph_policy(*, policy_graph, session, lattice_result, signatures_result,
                                    connection_graph=None, budget=csp.EndpointBudget()):
    if type(policy_graph) is not ImageDerivedPolicyBoundGraph:
        return legacy._refusal("provenance_mismatch", "joint_policy.identity")
    try:
        ImageDerivedPolicyBoundGraph(**legacy._fields(policy_graph))
    except _ERRORS as error:
        return legacy._refusal("provenance_mismatch", "joint_policy.structure", type(error).__name__)
    graph = graph_mod.replay_image_derived_graph(graph_result=policy_graph.graph_result,
                    session=session, lattice_result=lattice_result, signatures_result=signatures_result,
                    connection_graph=connection_graph, budget=budget)
    if isinstance(graph, placement.PlacementRefusal):
        return graph
    rebuilt = bind_image_derived_graph_policy(graph_result=graph)
    if isinstance(rebuilt, placement.PlacementRefusal):
        return rebuilt
    if rebuilt.result_sha256 != policy_graph.result_sha256:
        return legacy._refusal("provenance_mismatch", "joint_policy.source_replay")
    return policy_graph
