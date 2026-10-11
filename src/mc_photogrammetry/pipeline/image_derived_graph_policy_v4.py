"""Phase B policy-v4 wrapper: source replay, no bound geometry state."""
from __future__ import annotations

from dataclasses import dataclass

from mc_photogrammetry.pipeline import image_derived_graph as legacy
from mc_photogrammetry.pipeline import image_derived_graph_v3 as graph_mod
from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
from mc_photogrammetry.pipeline import image_derived_shift_domain_policy as policy_mod
from mc_photogrammetry.reconstruction import plane_graph, plane_placement as placement
from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp

IMAGE_DERIVED_GRAPH_POLICY_SCHEMA = "mc-photogrammetry-image-derived-graph-policy-v4"
IMAGE_DERIVED_GRAPH_POLICY_PRODUCER = "m9-j-image-derived-graph-policy-4"
GRAPH_POLICY_TERMINAL = "geometry_unavailable"


def _policy():
    value = policy_mod.IMAGE_DERIVED_SHIFT_DOMAIN_POLICY
    rebuilt = policy_mod.ImageDerivedShiftDomainPolicy(**legacy._fields(value))
    if rebuilt.to_record() != policy_mod._builtin_policy().to_record():
        raise ValueError("Phase B policy differs from unchanged built-in policy")
    return value


def _payload(value):
    return {"schema": value.schema, "producer": value.producer, "terminal": value.terminal,
            "status": value.status, "reason": value.reason, "detail": value.detail,
            "graph_sha256": value.graph_sha256, "policy_sha256": value.policy_sha256,
            "frame_ids": value.frame_ids, "max_shift_cells": value.max_shift_cells,
            "correlation_algorithm": value.correlation_algorithm}


@dataclass(frozen=True, slots=True)
class ImageDerivedPolicyGraphV4:
    schema: str
    producer: str
    terminal: str
    status: str
    reason: str
    detail: str
    graph_sha256: str
    policy_sha256: str
    frame_ids: tuple[str, ...]
    graph_result: graph_mod.ImageDerivedGraphResultV3
    policy: policy_mod.ImageDerivedShiftDomainPolicy
    max_shift_cells: int = 32
    correlation_algorithm: str = plane_graph.PLANE_CORRELATION_ALGORITHM
    result_sha256: str = ""

    def __post_init__(self):
        if (self.schema != IMAGE_DERIVED_GRAPH_POLICY_SCHEMA or self.producer != IMAGE_DERIVED_GRAPH_POLICY_PRODUCER
                or self.terminal != GRAPH_POLICY_TERMINAL or self.status != "unavailable"
                or type(self.graph_result) is not graph_mod.ImageDerivedGraphResultV3
                or type(self.policy) is not policy_mod.ImageDerivedShiftDomainPolicy):
            raise ValueError("unsupported Phase B policy wrapper identity/state")
        graph_mod.ImageDerivedGraphResultV3(**legacy._fields(self.graph_result))
        policy_mod.ImageDerivedShiftDomainPolicy(**legacy._fields(self.policy))
        if (self.policy.to_record() != _policy().to_record()
                or self.graph_sha256 != self.graph_result.result_sha256
                or self.policy_sha256 != self.policy.policy_sha256
                or self.frame_ids != self.graph_result.frame_ids
                or self.reason != self.graph_result.reason or self.detail != self.graph_result.detail
                or type(self.max_shift_cells) is not int or self.max_shift_cells != 32
                or self.graph_result.max_shift_cells != self.max_shift_cells
                or self.correlation_algorithm != self.policy.correlation_algorithm_id
                or self.graph_result.correlation_algorithm != self.correlation_algorithm):
            raise ValueError("Phase B policy wrapper bindings differ")
        digest = selection._sha(_payload(self))
        if self.result_sha256 not in ("", digest):
            raise ValueError("Phase B policy wrapper digest mismatch")
        object.__setattr__(self, "result_sha256", digest)


def bind_image_derived_graph_policy_v4(*, graph_result, session, lattice_result,
                                       signatures_result, connection_graph=None, budget=csp.EndpointBudget()):
    """Explicit validated-source replay; deliberately returns unavailable."""
    try:
        policy = _policy()
    except graph_mod._ERRORS as error:
        return legacy._refusal("provenance_mismatch", "phase_b.policy", type(error).__name__)
    graph = graph_mod.replay_image_derived_graph_v3(graph_result=graph_result, session=session,
                    lattice_result=lattice_result, signatures_result=signatures_result,
                    connection_graph=connection_graph, budget=budget)
    if isinstance(graph, placement.PlacementRefusal):
        return graph
    return ImageDerivedPolicyGraphV4(IMAGE_DERIVED_GRAPH_POLICY_SCHEMA, IMAGE_DERIVED_GRAPH_POLICY_PRODUCER,
                GRAPH_POLICY_TERMINAL, "unavailable", graph.reason, graph.detail,
                graph.result_sha256, policy.policy_sha256, graph.frame_ids, graph, policy)


def replay_image_derived_graph_policy_v4(*, policy_graph, session, lattice_result,
                                         signatures_result, connection_graph=None, budget=csp.EndpointBudget()):
    if type(policy_graph) is not ImageDerivedPolicyGraphV4:
        return legacy._refusal("provenance_mismatch", "phase_b.policy_identity")
    try:
        ImageDerivedPolicyGraphV4(**legacy._fields(policy_graph))
    except graph_mod._ERRORS as error:
        return legacy._refusal("provenance_mismatch", "phase_b.policy_structure", type(error).__name__)
    replayed = bind_image_derived_graph_policy_v4(graph_result=policy_graph.graph_result,
                    session=session, lattice_result=lattice_result, signatures_result=signatures_result,
                    connection_graph=connection_graph, budget=budget)
    if isinstance(replayed, placement.PlacementRefusal):
        return replayed
    if replayed.result_sha256 != policy_graph.result_sha256:
        return legacy._refusal("provenance_mismatch", "phase_b.policy_source_replay")
    return policy_graph
