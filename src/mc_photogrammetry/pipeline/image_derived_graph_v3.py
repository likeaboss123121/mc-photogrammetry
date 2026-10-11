"""Phase B source-backed endpoint selection; geometry is always unavailable.

This module has no production dispatch. Explicit replay requires the complete
validated source records and never invokes a geometry solver or evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass

from mc_photogrammetry.pipeline import image_derived_graph as legacy
from mc_photogrammetry.pipeline import image_derived_lattice_measure as lattice_mod
from mc_photogrammetry.pipeline import image_derived_signatures as signatures_mod
from mc_photogrammetry.pipeline import image_derived_shift_domain_policy as policy_mod
from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp
from mc_photogrammetry.reconstruction import lattice_planes, plane_graph, plane_placement as placement

IMAGE_DERIVED_GRAPH_SCHEMA = "mc-photogrammetry-image-derived-graph-v3"
IMAGE_DERIVED_GRAPH_PRODUCER = "m9-g-image-derived-graph-6"
GRAPH_TERMINAL = "geometry_unavailable"
_ERRORS = (*legacy._ERRORS, csp._Exhausted, RecursionError)
LEGACY_ENDPOINT_SELECTION_SCHEMA = "mc-photogrammetry-endpoint-selection-v1"
LEGACY_ENDPOINT_SELECTION_PRODUCER = "m9-g-endpoint-selection-1"


def _legacy_selection_identity(cert=None):
    schema = selection.ENDPOINT_SELECTION_SCHEMA if cert is None else getattr(cert, "schema", None)
    producer = selection.ENDPOINT_SELECTION_PRODUCER if cert is None else getattr(cert, "producer", None)
    return (schema, producer) == (LEGACY_ENDPOINT_SELECTION_SCHEMA, LEGACY_ENDPOINT_SELECTION_PRODUCER)


def _payload(result):
    return {"schema": result.schema, "producer": result.producer,
            "terminal": result.terminal, "status": result.status,
            "reason": result.reason, "detail": result.detail,
            "session_sha256": result.session_sha256, "lattice_sha256": result.lattice_sha256,
            "signatures_sha256": result.signatures_sha256, "connection_sha256": result.connection_sha256,
            "frame_ids": result.frame_ids, "reading_bindings": result.selection.reading_bindings,
            "signature_digests": [s.signature_sha256 for s in result.signatures],
            "candidate_matches": [legacy._match_payload(m) for m in result.candidate_matches],
            "selection_sha256": result.selection.result_sha256,
            "matches": [], "component_solution": None,
            "correlation_algorithm": result.correlation_algorithm, "max_shift_cells": result.max_shift_cells}


@dataclass(frozen=True, slots=True)
class ImageDerivedGraphResultV3:
    schema: str
    producer: str
    terminal: str
    status: str
    reason: str
    detail: str
    session_sha256: str
    lattice_sha256: str
    signatures_sha256: str
    connection_sha256: str
    frame_ids: tuple[str, ...]
    readings: tuple[lattice_planes.PlaneReading, ...]
    signatures: tuple[signatures_mod.SignatureEvidence, ...]
    candidate_matches: tuple[plane_graph.PlaneMatch, ...]
    selection: selection.EndpointSelectionCertificate
    matches: tuple = ()
    component_solution: None = None
    correlation_algorithm: str = plane_graph.PLANE_CORRELATION_ALGORITHM
    max_shift_cells: int = 32
    result_sha256: str = ""

    def __post_init__(self):
        if not _legacy_selection_identity(self.selection):
            raise ValueError("Phase B requires historical v1 selection identity")
        if (self.schema != IMAGE_DERIVED_GRAPH_SCHEMA or self.producer != IMAGE_DERIVED_GRAPH_PRODUCER
                or self.terminal != GRAPH_TERMINAL or self.status != "unavailable"
                or type(self.reason) is not str or not self.reason or type(self.detail) is not str
                or self.matches != () or type(self.matches) is not tuple or self.component_solution is not None
                or self.correlation_algorithm != plane_graph.PLANE_CORRELATION_ALGORITHM
                or type(self.max_shift_cells) is not int or self.max_shift_cells != 32
                or type(self.selection) is not selection.EndpointSelectionCertificate):
            raise ValueError("unsupported or non-atomic Phase B graph")
        selection._frames(self.frame_ids)
        selection.validate_candidate_certificate(self.selection, self.readings, self.signatures, self.candidate_matches)
        cert = self.selection
        if (self.session_sha256 != cert.session_sha256 or self.lattice_sha256 != cert.lattice_sha256
                or self.signatures_sha256 != cert.signatures_sha256 or self.connection_sha256 != cert.connection_sha256
                or self.frame_ids != cert.frame_ids or self.reason != cert.reason or self.detail != cert.status):
            raise ValueError("Phase B graph source/selection bindings differ")
        digest = selection._sha(_payload(self))
        if self.result_sha256 not in ("", digest):
            raise ValueError("Phase B graph digest mismatch")
        object.__setattr__(self, "result_sha256", digest)


def _sources(*, session, lattice_result, signatures_result, connection_graph):
    accepted = legacy._replay_session(session)
    if isinstance(accepted, placement.PlacementRefusal):
        return accepted
    lattice = legacy._replay(lattice_result, lattice_mod.ImageDerivedLatticeResult, "lattice_result")
    if isinstance(lattice, placement.PlacementRefusal):
        return lattice
    signatures = legacy._replay(signatures_result, signatures_mod.ImageDerivedSignaturesResult, "signatures_result")
    if isinstance(signatures, placement.PlacementRefusal):
        return signatures
    frames = tuple(f.frame_id for f in accepted.frames)
    if (lattice.session_sha256 != accepted.session_sha256 or signatures.session_sha256 != accepted.session_sha256
            or signatures.lattice_sha256 != lattice.result_sha256
            or lattice.frame_ids != frames or signatures.frame_ids != frames):
        return legacy._refusal("provenance_mismatch", "phase_b.upstream.binding")
    if lattice.status != "assembled" or signatures.status != "assembled":
        return legacy._refusal("invalid_input", "phase_b.upstream_unavailable")
    constraints = legacy._connection_constraints(connection_graph, accepted, lattice)
    if isinstance(constraints, placement.PlacementRefusal):
        return constraints
    readings, evidence = legacy._ordered_universe(lattice, signatures)
    return accepted, lattice, signatures, frames, readings, evidence, constraints


def derive_image_derived_graph_v3(*, session, lattice_result, signatures_result,
                                   connection_graph=None, budget=csp.EndpointBudget()):
    """One unchanged matcher pass and one Phase A pass; zero geometry calls."""
    if not _legacy_selection_identity():
        return legacy._refusal("provenance_mismatch", "phase_b.selection_identity")
    try:
        source = _sources(session=session, lattice_result=lattice_result,
                          signatures_result=signatures_result, connection_graph=connection_graph)
        if isinstance(source, placement.PlacementRefusal):
            return source
        accepted, lattice, signatures, frames, readings, evidence, constraints = source
        # Prebound the complete record universe before the matcher allocations.
        selection._frames(frames)
        selection.reading_bindings(readings, evidence)
        if type(budget) is not csp.EndpointBudget:
            raise ValueError("Phase B requires exact EndpointBudget")
        csp.EndpointBudget(**legacy._fields(budget))
        matches = plane_graph.match_plane_graph(readings, tuple(e.signature for e in evidence), max_shift_cells=32)
        cert = selection.select_endpoint_hypotheses(readings=readings, signatures=evidence,
                    candidate_matches=matches, frame_ids=frames, quadrant_constraints=constraints,
                    budget=budget, session_sha256=accepted.session_sha256,
                    lattice_sha256=lattice.result_sha256, signatures_sha256=signatures.result_sha256,
                    connection_sha256="" if connection_graph is None else connection_graph.result_sha256
                    if hasattr(connection_graph, "result_sha256") else connection_graph.graph_sha256)
        return ImageDerivedGraphResultV3(IMAGE_DERIVED_GRAPH_SCHEMA, IMAGE_DERIVED_GRAPH_PRODUCER,
                    GRAPH_TERMINAL, "unavailable", cert.reason, cert.status,
                    cert.session_sha256, cert.lattice_sha256, cert.signatures_sha256, cert.connection_sha256,
                    frames, readings, evidence, matches, cert)
    except _ERRORS as error:
        return legacy._refusal("provenance_mismatch", "phase_b.derivation", type(error).__name__)


def replay_image_derived_graph_v3(*, graph_result, session, lattice_result,
                                  signatures_result, connection_graph=None, budget=csp.EndpointBudget()):
    """Source replay is mandatory; a resealed graph hash cannot prove uniqueness."""
    if not _legacy_selection_identity() or not _legacy_selection_identity(getattr(graph_result, "selection", None)):
        return legacy._refusal("provenance_mismatch", "phase_b.selection_identity")
    if type(graph_result) is not ImageDerivedGraphResultV3:
        return legacy._refusal("provenance_mismatch", "phase_b.graph_identity")
    try:
        ImageDerivedGraphResultV3(**legacy._fields(graph_result))
        if type(budget) is not csp.EndpointBudget or graph_result.selection.budget != budget:
            return legacy._refusal("provenance_mismatch", "phase_b.expected_budget")
    except _ERRORS as error:
        return legacy._refusal("provenance_mismatch", "phase_b.graph_structure", type(error).__name__)
    replayed = derive_image_derived_graph_v3(session=session, lattice_result=lattice_result,
                    signatures_result=signatures_result, connection_graph=connection_graph,
                    budget=budget)
    if isinstance(replayed, placement.PlacementRefusal):
        return replayed
    if (replayed.result_sha256 != graph_result.result_sha256
            or selection._payload(replayed.selection) != selection._payload(graph_result.selection)
            or replayed.candidate_matches != graph_result.candidate_matches):
        return legacy._refusal("provenance_mismatch", "phase_b.source_replay")
    return graph_result
