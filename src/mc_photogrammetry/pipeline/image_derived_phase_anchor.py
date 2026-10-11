"""Derive V2-B phase anchors from accepted image and split-phase evidence."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

from mc_photogrammetry.pipeline import image_derived_endpoint_provenance as endpoint_mod
from mc_photogrammetry.pipeline import image_derived_lattice_measure as measure
from mc_photogrammetry.pipeline import image_derived_phase_anchor_policy as policy_mod
from mc_photogrammetry.pipeline import image_derived_relative_translation_policy as relative_policy_mod
from mc_photogrammetry.reconstruction import plane_geometry_provenance as provenance
from mc_photogrammetry.reconstruction import plane_phase_anchor
from mc_photogrammetry.reconstruction import plane_placement as placement

__all__ = ["IMAGE_DERIVED_PHASE_ANCHOR_SCHEMA",
           "IMAGE_DERIVED_PHASE_ANCHOR_PRODUCER",
           "ImageDerivedPhaseAnchorResult", "derive_image_derived_phase_anchor"]

IMAGE_DERIVED_PHASE_ANCHOR_SCHEMA = \
    "mc-photogrammetry-image-derived-phase-anchor"
IMAGE_DERIVED_PHASE_ANCHOR_PRODUCER = "m9-p2-image-derived-phase-anchor-1"
PHASE_ANCHOR_TERMINAL = "geometry_unavailable"
_ERRORS = (AttributeError, TypeError, ValueError, IndexError, KeyError,
           OverflowError, ZeroDivisionError, FloatingPointError)


def _sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _hex(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


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


def _replay_anchor(value: object) -> plane_phase_anchor.V2BAnchorResult:
    if not isinstance(value, plane_phase_anchor.V2BAnchorResult):
        raise ValueError("V2-B result has an invalid type")
    replayed = _deep_replay(value)
    if (not isinstance(replayed, plane_phase_anchor.V2BAnchorResult)
            or replayed.to_json_bytes() != value.to_json_bytes()):
        raise ValueError("V2-B result does not replay exactly")
    return replayed


def _payload(value: "ImageDerivedPhaseAnchorResult") -> dict:
    return {
        "schema": value.schema, "producer": value.producer,
        "terminal": value.terminal, "status": value.status,
        "reason": value.reason, "detail": value.detail,
        "endpoint_provenance_result_sha256":
            value.endpoint_provenance_result_sha256,
        "lattice_result_sha256": value.lattice_result_sha256,
        "graph_sha256": value.graph_sha256,
        "relative_policy_sha256": value.relative_policy_sha256,
        "endpoint_policy_sha256": value.endpoint_policy_sha256,
        "anchor_policy_sha256": value.anchor_policy_sha256,
        "calibration_sha256": value.calibration_sha256,
        "frame_ids": list(value.frame_ids),
        "anchor_result_sha256": value.anchor_result_sha256,
    }


@dataclass(frozen=True, slots=True)
class ImageDerivedPhaseAnchorResult:
    schema: str
    producer: str
    terminal: str
    status: str
    reason: str
    detail: str
    endpoint_provenance_result_sha256: str
    lattice_result_sha256: str
    graph_sha256: str
    relative_policy_sha256: str
    endpoint_policy_sha256: str
    anchor_policy_sha256: str
    calibration_sha256: str
    frame_ids: tuple[str, ...]
    endpoint_provenance_result: endpoint_mod.ImageDerivedEndpointProvenanceResult
    lattice_result: measure.ImageDerivedLatticeResult
    anchor_policy: policy_mod.ImageDerivedPhaseAnchorPolicy
    phase_anchor_result: plane_phase_anchor.V2BAnchorResult | None
    anchor_result_sha256: str
    result_sha256: str = ""

    def __post_init__(self) -> None:
        if (self.schema != IMAGE_DERIVED_PHASE_ANCHOR_SCHEMA
                or self.producer != IMAGE_DERIVED_PHASE_ANCHOR_PRODUCER
                or self.terminal != PHASE_ANCHOR_TERMINAL
                or self.status not in ("determined", "partial", "abstained",
                                       "refused", "budget_exhausted", "unavailable")
                or type(self.reason) is not str or type(self.detail) is not str
                or not all(_hex(item) for item in (
                    self.endpoint_provenance_result_sha256,
                    self.lattice_result_sha256, self.graph_sha256,
                    self.relative_policy_sha256, self.endpoint_policy_sha256,
                    self.anchor_policy_sha256, self.calibration_sha256))
                or not isinstance(self.frame_ids, tuple) or not self.frame_ids
                or not isinstance(self.endpoint_provenance_result,
                                  endpoint_mod.ImageDerivedEndpointProvenanceResult)
                or not isinstance(self.lattice_result,
                                  measure.ImageDerivedLatticeResult)
                or not isinstance(self.anchor_policy,
                                  policy_mod.ImageDerivedPhaseAnchorPolicy)):
            raise ValueError("image-derived phase anchor is malformed")
        endpoint = endpoint_mod.ImageDerivedEndpointProvenanceResult(**{
            field.name: getattr(self.endpoint_provenance_result, field.name)
            for field in dataclasses.fields(
                endpoint_mod.ImageDerivedEndpointProvenanceResult)})
        lattice = measure.ImageDerivedLatticeResult(**{
            field.name: getattr(self.lattice_result, field.name)
            for field in dataclasses.fields(measure.ImageDerivedLatticeResult)})
        policy = policy_mod.ImageDerivedPhaseAnchorPolicy(**{
            field.name: getattr(self.anchor_policy, field.name)
            for field in dataclasses.fields(policy_mod.ImageDerivedPhaseAnchorPolicy)})
        if (self.endpoint_provenance_result_sha256 != endpoint.result_sha256
                or self.lattice_result_sha256 != lattice.result_sha256
                or self.graph_sha256 != endpoint.graph_sha256
                or self.relative_policy_sha256 != endpoint.relative_policy_sha256
                or self.endpoint_policy_sha256 != endpoint.endpoint_policy_sha256
                or self.anchor_policy_sha256 != policy.policy_sha256
                or self.calibration_sha256 != policy.calibration_sha256
                or policy.to_record()
                    != policy_mod.IMAGE_DERIVED_PHASE_ANCHOR_POLICY.to_record()
                or self.frame_ids != endpoint.frame_ids):
            raise ValueError("phase-anchor sources do not bind")
        if self.status == "unavailable":
            if (not self.reason or self.phase_anchor_result is not None
                    or self.anchor_result_sha256):
                raise ValueError("unavailable phase anchor must be atomic")
        else:
            result = _replay_anchor(self.phase_anchor_result)
            if (self.anchor_result_sha256 != result.result_sha256
                    or result.status != self.status):
                raise ValueError("V2-B result does not bind")
            if result.prediction is not None:
                prediction = result.prediction
                relative = endpoint.relative_translations_result
                vector = relative.vector_domains_result.vector_domain_result
                v2a = endpoint.endpoint_provenance_result
                if (relative.relative_translation_result is None or vector is None
                        or v2a is None
                        or prediction.c2rb_result_sha256
                            != relative.relative_translation_result.result_sha256
                        or prediction.vector_result_sha256 != vector.result_sha256
                        or prediction.v2a_result_sha256 != v2a.result_sha256
                        or prediction.calibration_sha256 != policy.calibration_sha256
                        or prediction.budget_sha256 != policy.budget.budget_sha256):
                    raise ValueError("V2-B prediction provenance does not bind")
            if self.status in ("determined", "partial", "abstained"):
                if self.reason or self.detail:
                    raise ValueError("V2-B prediction cannot carry wrapper refusal")
            elif not self.reason:
                raise ValueError("V2-B non-prediction needs a reason")
        digest = _sha(_payload(self))
        if self.result_sha256 not in ("", digest):
            raise ValueError("image-derived phase-anchor digest mismatch")
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


def _make(endpoint, lattice, *, status, reason="", detail="", result=None):
    policy = policy_mod.IMAGE_DERIVED_PHASE_ANCHOR_POLICY
    return ImageDerivedPhaseAnchorResult(
        IMAGE_DERIVED_PHASE_ANCHOR_SCHEMA, IMAGE_DERIVED_PHASE_ANCHOR_PRODUCER,
        PHASE_ANCHOR_TERMINAL, status, reason, detail, endpoint.result_sha256,
        lattice.result_sha256, endpoint.graph_sha256,
        endpoint.relative_policy_sha256, endpoint.endpoint_policy_sha256,
        policy.policy_sha256, policy.calibration_sha256, endpoint.frame_ids,
        endpoint, lattice, policy, result,
        "" if result is None else result.result_sha256)


def derive_image_derived_phase_anchor(
    *, endpoint_provenance_result: object, lattice_result: object,
) -> ImageDerivedPhaseAnchorResult | placement.PlacementRefusal:
    """Run V2-B once with exact O2 evidence and sealed P0 split sources."""
    endpoint = _replay(endpoint_provenance_result,
                       endpoint_mod.ImageDerivedEndpointProvenanceResult,
                       "endpoint_provenance_result")
    if isinstance(endpoint, placement.PlacementRefusal):
        return endpoint
    lattice = _replay(lattice_result, measure.ImageDerivedLatticeResult,
                      "lattice_result")
    if isinstance(lattice, placement.PlacementRefusal):
        return lattice
    if endpoint.status != "complete":
        return _make(endpoint, lattice, status="unavailable",
                     reason="endpoint_provenance_" + endpoint.status,
                     detail=endpoint.reason)
    if lattice.status != "assembled" or lattice.bridge is None:
        return _refusal("provenance_mismatch", "lattice_result.ready_state")
    relative = endpoint.relative_translations_result
    vector_wrapper = relative.vector_domains_result
    shift = vector_wrapper.shift_domains_result
    phase = vector_wrapper.phase_evidence_result
    graph = shift.policy_graph_result.graph_result
    vector = vector_wrapper.vector_domain_result
    closure = relative.relative_translation_result
    v2a = endpoint.endpoint_provenance_result
    try:
        if (graph.lattice_sha256 != lattice.result_sha256
                or lattice.frame_ids != endpoint.frame_ids
                or lattice.bridge.reading_universe_sha256
                    != vector.canonical_reading_universe_sha256
                or len(lattice.readings) != len(graph.readings)
                or any(left is not right for left, right in zip(
                    lattice.readings, graph.readings))):
            raise ValueError("lattice bridge differs from graph universe")
        rebuilt = provenance.bind_geometry_provenance(
            readings=lattice.readings,
            structural_weights=lattice.bridge.structural_weights,
            split_sources=lattice.bridge.split_sources)
        if (not isinstance(rebuilt, provenance.GeometryProvenanceBridge)
                or rebuilt.bridge_sha256 != lattice.bridge.bridge_sha256):
            raise ValueError("lattice bridge does not replay")
        split_sources = lattice.bridge.split_sources_for_assembly()
        support = shift.axis_support_result.axis_support
        if (vector is None or closure is None or v2a is None or support is None
                or shift.shift_domain_result is None):
            raise ValueError("accepted upstream evidence is incomplete")
        policy = policy_mod.IMAGE_DERIVED_PHASE_ANCHOR_POLICY
        relative_policy = relative.relative_policy
        calibration = plane_phase_anchor.V2B_BUILTIN_CALIBRATION
        if (relative_policy.to_record()
                != relative_policy_mod.IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY.to_record()
                or relative_policy.comparison_tolerance
                    != calibration.replay_comparison_tolerance
                or relative_policy.closure_tolerance
                    != calibration.replay_closure_tolerance):
            raise ValueError("V2-B replay tolerances differ from calibration")
        result = plane_phase_anchor.analyze_phase_anchor(
            closure, vector, support, shift.shift_domain_result, graph.matches,
            phase.phase_evidence, readings=graph.readings,
            signatures=tuple(item.signature for item in graph.signatures),
            comparison_tolerance=relative_policy.comparison_tolerance,
            closure_tolerance=relative_policy.closure_tolerance,
            v2a_result=v2a, split_sources=split_sources,
            budget=policy.budget, calibration=calibration)
        replayed = _replay_anchor(result)
    except _ERRORS as error:
        return _refusal("provenance_mismatch", "phase_anchor_result",
                        type(error).__name__)
    if result.status in ("determined", "partial", "abstained"):
        return _make(endpoint, lattice, status=result.status, result=result)
    if result.status == "refused":
        detail = ",".join(item.reason for item in result.refusals)
        return _make(endpoint, lattice, status="refused",
                     reason="phase_anchor_refused", detail=detail, result=result)
    if result.status == "budget_exhausted":
        return _make(endpoint, lattice, status="budget_exhausted",
                     reason="phase_anchor_budget_exhausted", result=result)
    return _refusal("provenance_mismatch", "phase_anchor_result.status")
