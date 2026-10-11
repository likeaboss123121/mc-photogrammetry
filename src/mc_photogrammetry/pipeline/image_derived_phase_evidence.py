"""Derive exact P1-B phase evidence from an accepted policy-bound graph."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

from mc_photogrammetry.pipeline import image_derived_axis_support as axis_adapter
from mc_photogrammetry.pipeline import image_derived_graph_policy_v5 as graph_policy_mod
from mc_photogrammetry.reconstruction import lattice_planes, plane_residue
from mc_photogrammetry.reconstruction import plane_placement as placement

__all__ = ["IMAGE_DERIVED_PHASE_EVIDENCE_SCHEMA",
           "IMAGE_DERIVED_PHASE_EVIDENCE_PRODUCER",
           "ImageDerivedPhaseEvidenceResult", "derive_image_derived_phase_evidence"]

IMAGE_DERIVED_PHASE_EVIDENCE_SCHEMA = "mc-photogrammetry-image-derived-phase-evidence"
IMAGE_DERIVED_PHASE_EVIDENCE_PRODUCER = "m9-l-image-derived-phase-evidence-1"
PHASE_EVIDENCE_TERMINAL = "geometry_unavailable"
_ERRORS = (AttributeError, TypeError, ValueError, IndexError, KeyError,
           OverflowError, ZeroDivisionError, FloatingPointError)


def _sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _hex(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _phase_sha(values: tuple[plane_residue.HypothesisPhaseShiftEvidence, ...]) -> str:
    return _sha([value.to_record() for value in values])


def _payload(value: "ImageDerivedPhaseEvidenceResult") -> dict:
    return {
        "schema": value.schema, "producer": value.producer,
        "terminal": value.terminal, "status": value.status,
        "reason": value.reason, "detail": value.detail,
        "policy_graph_result_sha256": value.policy_graph_result_sha256,
        "graph_sha256": value.graph_sha256,
        "policy_sha256": value.policy_sha256,
        "max_shift_cells": value.max_shift_cells,
        "frame_ids": list(value.frame_ids),
        "phase_evidence_sha256": value.phase_evidence_sha256,
    }


def _fields(value: object) -> dict:
    return {field.name: getattr(value, field.name)
            for field in dataclasses.fields(value)}


def _replay_hypothesis(
    value: object,
) -> plane_residue.HypothesisPhaseShiftEvidence:
    if not isinstance(value, plane_residue.HypothesisPhaseShiftEvidence):
        raise ValueError("phase evidence has an invalid type")
    candidates = []
    for candidate in value.phase_candidates:
        if not isinstance(candidate, plane_residue.PeriodicResidueCandidates):
            raise ValueError("phase candidate has an invalid type")
        quality = candidate.quality
        if quality is not None:
            if not isinstance(quality, plane_residue.PhaseAxisQuality):
                raise ValueError("phase quality has an invalid type")
            quality = plane_residue.PhaseAxisQuality(**_fields(quality))
        candidate_fields = _fields(candidate)
        candidate_fields["quality"] = quality
        candidates.append(plane_residue.PeriodicResidueCandidates(
            **candidate_fields))
    support = value.relative_shift
    if not isinstance(support, plane_residue.RelativeShiftSupport):
        raise ValueError("relative shift has an invalid type")
    fields = _fields(value)
    fields["phase_candidates"] = tuple(candidates)
    fields["relative_shift"] = plane_residue.RelativeShiftSupport(
        **_fields(support))
    replayed = plane_residue.HypothesisPhaseShiftEvidence(**fields)
    if replayed.to_record() != value.to_record():
        raise ValueError("phase evidence does not replay exactly")
    return replayed


def _identity(value: plane_residue.HypothesisPhaseShiftEvidence) -> tuple:
    return (value.left_frame, value.right_frame, value.left_plane,
            value.right_plane, value.left_axis, value.right_axis,
            value.left_b, value.right_b, value.quadrant_of_right)


def _match_identity(match: object) -> tuple:
    return (match.left, match.right, match.left_plane, match.right_plane,
            match.left_axis, match.right_axis, match.left_b, match.right_b,
            match.quadrant_of_right)


def _bind_to_match(value: plane_residue.HypothesisPhaseShiftEvidence,
                   match: object, readings: dict) -> None:
    if _identity(value) != _match_identity(match):
        raise ValueError("phase evidence identity is foreign to its match")
    support = value.relative_shift
    left = readings[(match.left, match.left_plane)]
    right = readings[(match.right, match.right_plane)]
    expected = (
        match.cell_shift_rc, match.psr, match.peak, match.second_peak,
        match.overlap_cells, left.structural_share, right.structural_share,
        match.score)
    actual = (
        support.cell_shift_rc,
        plane_residue._psr_from_record(support.to_record()["psr"]),
        support.peak, support.second_peak,
        support.overlap_cells, support.left_structural_share,
        support.right_structural_share, support.score)
    if actual != expected:
        raise ValueError("phase relative-shift support differs from its match")


@dataclass(frozen=True, slots=True)
class ImageDerivedPhaseEvidenceResult:
    schema: str
    producer: str
    terminal: str
    status: str
    reason: str
    detail: str
    policy_graph_result_sha256: str
    graph_sha256: str
    policy_sha256: str
    max_shift_cells: int
    frame_ids: tuple[str, ...]
    policy_graph_result: graph_policy_mod.ImageDerivedPolicyBoundGraph
    phase_evidence: tuple[plane_residue.HypothesisPhaseShiftEvidence, ...]
    phase_evidence_sha256: str
    result_sha256: str = ""

    def __post_init__(self) -> None:
        if (self.schema != IMAGE_DERIVED_PHASE_EVIDENCE_SCHEMA
                or self.producer != IMAGE_DERIVED_PHASE_EVIDENCE_PRODUCER
                or self.terminal != PHASE_EVIDENCE_TERMINAL
                or self.status not in ("complete", "refused", "unavailable")
                or type(self.reason) is not str or type(self.detail) is not str
                or not all(_hex(item) for item in (
                    self.policy_graph_result_sha256, self.graph_sha256,
                    self.policy_sha256, self.phase_evidence_sha256))
                or type(self.max_shift_cells) is not int
                or self.max_shift_cells <= 0
                or not isinstance(self.frame_ids, tuple) or not self.frame_ids
                or len(set(self.frame_ids)) != len(self.frame_ids)
                or any(type(item) is not str or not item for item in self.frame_ids)
                or not isinstance(self.policy_graph_result,
                                  graph_policy_mod.ImageDerivedPolicyBoundGraph)
                or not isinstance(self.phase_evidence, tuple)):
            raise ValueError("image-derived phase-evidence result is malformed")
        graph = graph_policy_mod.ImageDerivedPolicyBoundGraph(**{
            field.name: getattr(self.policy_graph_result, field.name)
            for field in dataclasses.fields(
                graph_policy_mod.ImageDerivedPolicyBoundGraph)})
        if self.status != "unavailable":
            graph_policy_mod._require_live_policy(self.policy_graph_result)
        if (self.policy_graph_result_sha256 != graph.result_sha256
                or self.graph_sha256 != graph.graph_sha256
                or self.policy_sha256 != graph.policy_sha256
                or self.max_shift_cells != graph.max_shift_cells
                or self.frame_ids != graph.frame_ids):
            raise ValueError("phase-evidence source does not bind")
        replayed = tuple(_replay_hypothesis(item) for item in self.phase_evidence)
        if self.phase_evidence_sha256 != _phase_sha(replayed):
            raise ValueError("phase-evidence collection digest mismatch")
        if self.status == "unavailable":
            if not self.reason or self.phase_evidence:
                raise ValueError("unavailable phase evidence must be atomic")
        else:
            matches = graph.graph_result.matches
            if len(replayed) != len(matches) or not replayed:
                raise ValueError("phase evidence does not cover every match")
            indexed, _reverse = axis_adapter._reading_index(
                graph.graph_result.readings)
            for evidence, match in zip(replayed, matches):
                _bind_to_match(evidence, match, indexed)
            refused = tuple(item.refusal for item in replayed
                            if item.refusal is not None)
            if self.status == "complete" and (self.reason or self.detail or refused):
                raise ValueError("complete phase evidence is inconsistent")
            if self.status == "refused" and (
                    self.reason != "phase_evidence_refused" or not refused):
                raise ValueError("refused phase evidence is inconsistent")
        digest = _sha(_payload(self))
        if self.result_sha256 not in ("", digest):
            raise ValueError("image-derived phase-evidence digest mismatch")
        object.__setattr__(self, "result_sha256", digest)


def _refusal(reason: str, field: str, error: object = "") -> placement.PlacementRefusal:
    detail = {"field": field}
    if error:
        detail["detail"] = str(error)
    return placement._make_refusal(reason, detail)


def _replay_graph(value: object) -> graph_policy_mod.ImageDerivedPolicyBoundGraph | placement.PlacementRefusal:
    if not isinstance(value, graph_policy_mod.ImageDerivedPolicyBoundGraph):
        return _refusal("invalid_input", "policy_graph_result")
    try:
        graph_policy_mod.ImageDerivedPolicyBoundGraph(**{
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(
                graph_policy_mod.ImageDerivedPolicyBoundGraph)})
        graph_policy_mod._require_live_policy(value)
    except _ERRORS as error:
        return _refusal("provenance_mismatch", "policy_graph_result",
                        type(error).__name__)
    return value


def _make(graph: graph_policy_mod.ImageDerivedPolicyBoundGraph, *,
          status: str, reason: str = "", detail: str = "",
          evidence: tuple = ()) -> ImageDerivedPhaseEvidenceResult:
    return ImageDerivedPhaseEvidenceResult(
        IMAGE_DERIVED_PHASE_EVIDENCE_SCHEMA,
        IMAGE_DERIVED_PHASE_EVIDENCE_PRODUCER, PHASE_EVIDENCE_TERMINAL,
        status, reason, detail, graph.result_sha256, graph.graph_sha256,
        graph.policy_sha256, graph.max_shift_cells, graph.frame_ids, graph,
        evidence, _phase_sha(evidence))


def derive_image_derived_phase_evidence(
    *, policy_graph_result: object,
) -> ImageDerivedPhaseEvidenceResult | placement.PlacementRefusal:
    """Derive one exact P1-B hypothesis record per bound graph match."""
    graph = _replay_graph(policy_graph_result)
    if isinstance(graph, placement.PlacementRefusal):
        return graph
    if graph.status != "bound":
        return _make(graph, status="unavailable",
                     reason="policy_graph_unavailable", detail=graph.reason)
    try:
        readings, _evidence, matches, _solution, _raw = \
            axis_adapter._bind_universe(graph.graph_result)
        indexed, _reverse = axis_adapter._reading_index(readings)
        produced = []
        for match in matches:
            value = plane_residue.hypothesis_phase_shift_evidence(match, indexed)
            replayed = _replay_hypothesis(value)
            _bind_to_match(replayed, match, indexed)
            produced.append(value)
    except _ERRORS as error:
        return _refusal("provenance_mismatch", "phase_evidence",
                        type(error).__name__)
    values = tuple(produced)
    refusals = tuple(item.refusal for item in values if item.refusal is not None)
    if refusals:
        return _make(graph, status="refused", reason="phase_evidence_refused",
                     detail=",".join(refusals), evidence=values)
    return _make(graph, status="complete", evidence=values)
