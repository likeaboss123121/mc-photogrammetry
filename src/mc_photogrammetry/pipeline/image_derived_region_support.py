"""Build exact aligned V1 supports from accepted image-derived evidence."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

from mc_photogrammetry.pipeline import image_derived_phase_anchor as anchor_mod
from mc_photogrammetry.pipeline import image_derived_region_support_policy as policy_mod
from mc_photogrammetry.pipeline import image_derived_signatures as signature_mod
from mc_photogrammetry.reconstruction import lattice_planes
from mc_photogrammetry.reconstruction import plane_geometry_provenance as provenance
from mc_photogrammetry.reconstruction import plane_placement as placement
from mc_photogrammetry.reconstruction import plane_region_support

__all__ = ["IMAGE_DERIVED_REGION_SUPPORT_SCHEMA",
           "IMAGE_DERIVED_REGION_SUPPORT_PRODUCER", "RegionSupportEvidence",
           "ImageDerivedRegionSupportResult", "derive_image_derived_region_support"]

IMAGE_DERIVED_REGION_SUPPORT_SCHEMA = \
    "mc-photogrammetry-image-derived-region-support"
IMAGE_DERIVED_REGION_SUPPORT_PRODUCER = "m9-q2-image-derived-region-support-1"
REGION_SUPPORT_TERMINAL = "geometry_unavailable"
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


@dataclass(frozen=True, slots=True)
class RegionSupportEvidence:
    reading_identity_sha256: str
    signature_sha256: str
    usable_mask_sha256: str
    image_sha256: str
    support: plane_region_support.AlignedRegionSupport
    evidence_sha256: str = ""

    def __post_init__(self) -> None:
        if (not all(_hex(item) for item in (
                self.reading_identity_sha256, self.signature_sha256,
                self.usable_mask_sha256, self.image_sha256))
                or not isinstance(self.support,
                                  plane_region_support.AlignedRegionSupport)):
            raise ValueError("region-support evidence is malformed")
        support = _deep_replay(self.support)
        if (support.reading_identity_sha256 != self.reading_identity_sha256
                or support.image_sha256 != self.image_sha256
                or support.usable_mask_sha256 != self.usable_mask_sha256):
            raise ValueError("region-support evidence does not bind")
        digest = _sha([self.reading_identity_sha256, self.signature_sha256,
                       self.usable_mask_sha256, self.image_sha256,
                       support.support_sha256])
        if self.evidence_sha256 not in ("", digest):
            raise ValueError("region-support evidence digest mismatch")
        object.__setattr__(self, "evidence_sha256", digest)


def _payload(value: "ImageDerivedRegionSupportResult") -> dict:
    return {
        "schema": value.schema, "producer": value.producer,
        "terminal": value.terminal, "status": value.status,
        "reason": value.reason, "detail": value.detail,
        "phase_anchor_result_sha256": value.phase_anchor_result_sha256,
        "signatures_result_sha256": value.signatures_result_sha256,
        "session_sha256": value.session_sha256,
        "lattice_sha256": value.lattice_sha256,
        "graph_sha256": value.graph_sha256,
        "policy_sha256": value.policy_sha256,
        "phase_anchor_sha256": value.phase_anchor_sha256,
        "frame_ids": list(value.frame_ids),
        "support_evidence_sha256": [item.evidence_sha256
                                    for item in value.support_evidence],
        "refusal_sha256": [item.refusal_sha256 for item in value.refusals],
    }


@dataclass(frozen=True, slots=True)
class ImageDerivedRegionSupportResult:
    schema: str
    producer: str
    terminal: str
    status: str
    reason: str
    detail: str
    phase_anchor_result_sha256: str
    signatures_result_sha256: str
    session_sha256: str
    lattice_sha256: str
    graph_sha256: str
    policy_sha256: str
    phase_anchor_sha256: str
    frame_ids: tuple[str, ...]
    phase_anchor_result: anchor_mod.ImageDerivedPhaseAnchorResult
    signatures_result: signature_mod.ImageDerivedSignaturesResult
    policy: policy_mod.ImageDerivedRegionSupportPolicy
    support_evidence: tuple[RegionSupportEvidence, ...]
    refusals: tuple[plane_region_support.AlignedSupportRefusal, ...]
    result_sha256: str = ""

    def __post_init__(self) -> None:
        if (self.schema != IMAGE_DERIVED_REGION_SUPPORT_SCHEMA
                or self.producer != IMAGE_DERIVED_REGION_SUPPORT_PRODUCER
                or self.terminal != REGION_SUPPORT_TERMINAL
                or self.status not in (
                    "assembled", "refused", "budget_exhausted", "unavailable")
                or type(self.reason) is not str or type(self.detail) is not str
                or not all(_hex(item) for item in (
                    self.phase_anchor_result_sha256,
                    self.signatures_result_sha256, self.session_sha256,
                    self.lattice_sha256, self.graph_sha256, self.policy_sha256))
                or not isinstance(self.frame_ids, tuple) or not self.frame_ids
                or not isinstance(self.phase_anchor_result,
                                  anchor_mod.ImageDerivedPhaseAnchorResult)
                or not isinstance(self.signatures_result,
                                  signature_mod.ImageDerivedSignaturesResult)
                or not isinstance(self.policy,
                                  policy_mod.ImageDerivedRegionSupportPolicy)
                or not isinstance(self.support_evidence, tuple)
                or not isinstance(self.refusals, tuple)):
            raise ValueError("image-derived region supports are malformed")
        anchor = anchor_mod.ImageDerivedPhaseAnchorResult(**{
            field.name: getattr(self.phase_anchor_result, field.name)
            for field in dataclasses.fields(anchor_mod.ImageDerivedPhaseAnchorResult)})
        signatures = signature_mod.ImageDerivedSignaturesResult(**{
            field.name: getattr(self.signatures_result, field.name)
            for field in dataclasses.fields(signature_mod.ImageDerivedSignaturesResult)})
        policy = policy_mod.ImageDerivedRegionSupportPolicy(**{
            field.name: getattr(self.policy, field.name)
            for field in dataclasses.fields(policy_mod.ImageDerivedRegionSupportPolicy)})
        evidence = tuple(_deep_replay(item) for item in self.support_evidence)
        refusals = tuple(_deep_replay(item) for item in self.refusals)
        if (self.phase_anchor_result_sha256 != anchor.result_sha256
                or self.signatures_result_sha256 != signatures.result_sha256
                or self.session_sha256 != signatures.session_sha256
                or self.lattice_sha256 != signatures.lattice_sha256
                or self.graph_sha256 != anchor.graph_sha256
                or self.policy_sha256 != policy.policy_sha256
                or policy.to_record()
                    != policy_mod.IMAGE_DERIVED_REGION_SUPPORT_POLICY.to_record()
                or self.frame_ids != signatures.frame_ids):
            raise ValueError("region-support sources do not bind")
        if self.status == "assembled":
            if (self.reason or self.detail or refusals or not evidence
                    or not _hex(self.phase_anchor_sha256)):
                raise ValueError("assembled region supports are inconsistent")
            prediction = anchor.phase_anchor_result.prediction
            if (prediction is None or prediction.status != "determined"
                    or None in (prediction.alpha_x, prediction.alpha_y,
                                prediction.alpha_z)
                    or self.phase_anchor_sha256 != placement.placement_anchor_digest(
                        (prediction.alpha_x, prediction.alpha_y,
                         prediction.alpha_z))):
                raise ValueError("region supports do not bind the phase anchor")
            v2a_result = anchor.endpoint_provenance_result.endpoint_provenance_result
            if (v2a_result is None
                    or len(evidence) != len(v2a_result.reading_domains)):
                raise ValueError("region supports do not cover V2-A domains")
            masks = {item.frame_id: item for item in signatures.usable_masks}
            signature_index = {
                (item.frame_id, item.canonical_index, item.reading_axis,
                 item.blocks_per_period): item
                for item in signatures.signatures}
            for item, domain in zip(evidence, v2a_result.reading_domains):
                identity = domain.reading_identity
                signature = signature_index.get((
                    identity.frame, identity.canonical_index,
                    identity.reading_axis, domain.selected_b))
                mask = masks.get(identity.frame)
                if (signature is None or mask is None
                        or item.reading_identity_sha256
                            != identity.identity_sha256
                        or item.signature_sha256 != signature.signature_sha256
                        or item.usable_mask_sha256 != mask.mask_sha256
                        or item.support.phase_anchor_sha256
                            != self.phase_anchor_sha256
                        or item.support.alternative.blocks_per_period
                            != domain.selected_b):
                    raise ValueError("region-support evidence does not bind")
        elif self.status == "unavailable":
            if (not self.reason or evidence or refusals
                    or self.phase_anchor_sha256):
                raise ValueError("unavailable region supports must be atomic")
        elif (not self.reason or evidence or not refusals
              or self.phase_anchor_sha256):
            raise ValueError("failed region supports must be atomic")
        digest = _sha(_payload(self))
        if self.result_sha256 not in ("", digest):
            raise ValueError("image-derived region-support digest mismatch")
        object.__setattr__(self, "result_sha256", digest)


def _refusal(reason, field, error=""):
    detail = {"field": field}
    if error:
        detail["detail"] = str(error)
    return placement._make_refusal(reason, detail)


def _make(anchor, signatures, *, status, reason="", detail="", evidence=(),
          refusals=(), anchor_sha=""):
    policy = policy_mod.IMAGE_DERIVED_REGION_SUPPORT_POLICY
    return ImageDerivedRegionSupportResult(
        IMAGE_DERIVED_REGION_SUPPORT_SCHEMA,
        IMAGE_DERIVED_REGION_SUPPORT_PRODUCER, REGION_SUPPORT_TERMINAL,
        status, reason, detail, anchor.result_sha256, signatures.result_sha256,
        signatures.session_sha256, signatures.lattice_sha256,
        anchor.graph_sha256, policy.policy_sha256, anchor_sha,
        signatures.frame_ids, anchor, signatures, policy,
        tuple(evidence), tuple(refusals))


def derive_image_derived_region_support(
    *, phase_anchor_result: object, signatures_result: object, session: object,
) -> ImageDerivedRegionSupportResult | placement.PlacementRefusal:
    """Build one exact V1 support per canonical V2-A reading domain."""
    if not isinstance(phase_anchor_result, anchor_mod.ImageDerivedPhaseAnchorResult):
        return _refusal("invalid_input", "phase_anchor_result")
    if not isinstance(signatures_result, signature_mod.ImageDerivedSignaturesResult):
        return _refusal("invalid_input", "signatures_result")
    try:
        anchor = anchor_mod.ImageDerivedPhaseAnchorResult(**{
            field.name: getattr(phase_anchor_result, field.name)
            for field in dataclasses.fields(anchor_mod.ImageDerivedPhaseAnchorResult)})
        signatures = signature_mod.ImageDerivedSignaturesResult(**{
            field.name: getattr(signatures_result, field.name)
            for field in dataclasses.fields(signature_mod.ImageDerivedSignaturesResult)})
    except _ERRORS as error:
        return _refusal("provenance_mismatch", "upstream", type(error).__name__)
    replayed_session = signature_mod._replay_session(session)
    if isinstance(replayed_session, placement.PlacementRefusal):
        return replayed_session
    session_record, live_frames = replayed_session
    if (session_record.session_sha256 != signatures.session_sha256
            or tuple(frame.frame_id for frame in live_frames) != signatures.frame_ids):
        return _refusal("provenance_mismatch", "session.binding")
    if anchor.status != "determined":
        return _make(anchor, signatures, status="unavailable",
                     reason="phase_anchor_" + anchor.status,
                     detail=anchor.reason)
    if (signatures.status != "assembled"
            or anchor.lattice_result_sha256 != signatures.lattice_sha256):
        return _refusal("provenance_mismatch", "signatures.binding")
    endpoint = anchor.endpoint_provenance_result
    v2a = endpoint.endpoint_provenance_result
    prediction = anchor.phase_anchor_result.prediction
    if v2a is None or prediction is None or prediction.status != "determined":
        return _refusal("provenance_mismatch", "phase_anchor.ready_state")
    alphas = (prediction.alpha_x, prediction.alpha_y, prediction.alpha_z)
    if any(item is None for item in alphas):
        return _refusal("provenance_mismatch", "phase_anchor.alpha")
    anchor_sha = placement.placement_anchor_digest(alphas)
    graph = endpoint.relative_translations_result.vector_domains_result \
        .shift_domains_result.policy_graph_result.graph_result
    lattice = anchor.lattice_result
    bridge = lattice.bridge
    if (lattice.status != "assembled" or bridge is None
            or graph.lattice_sha256 != lattice.result_sha256
            or len(lattice.readings) != len(graph.readings)
            or any(left is not right for left, right in zip(
                lattice.readings, graph.readings))):
        return _refusal("provenance_mismatch", "lattice_bridge.binding")
    rebuilt_bridge = provenance.bind_geometry_provenance(
        readings=lattice.readings,
        structural_weights=bridge.structural_weights,
        split_sources=bridge.split_sources)
    if (not isinstance(rebuilt_bridge, provenance.GeometryProvenanceBridge)
            or rebuilt_bridge.bridge_sha256 != bridge.bridge_sha256):
        return _refusal("provenance_mismatch", "lattice_bridge.replay")
    masks = {item.frame_id: item for item in signatures.usable_masks}
    frames = {item.frame_id: item for item in live_frames}
    reading_index = {}
    by_frame = {}
    for reading in graph.readings:
        by_frame.setdefault(reading.frame, []).append(reading)
    for frame, values in by_frame.items():
        for index, reading in enumerate(sorted(values,
                                               key=lattice_planes.reading_order_key)):
            reading_index[(frame, index, reading.axis)] = reading
    policy = policy_mod.IMAGE_DERIVED_REGION_SUPPORT_POLICY
    produced, refused = [], []
    try:
        for domain in v2a.reading_domains:
            identity = domain.reading_identity
            key = (identity.frame, identity.canonical_index,
                   identity.reading_axis)
            reading = reading_index.get(key)
            choices = [item for item in signatures.signatures
                       if (item.frame_id, item.canonical_index,
                           item.reading_axis) == key
                       and item.blocks_per_period == domain.selected_b]
            mask = masks.get(identity.frame)
            frame = frames.get(identity.frame)
            if reading is None or len(choices) != 1 or mask is None or frame is None:
                raise ValueError("support source join is incomplete")
            choice = choices[0]
            result = plane_region_support.build_aligned_region_support(
                reading=reading, reading_identity=identity,
                reading_universe=graph.readings, usable_mask=mask.mask,
                image_sha256=frame.image_sha256,
                blocks_per_period=domain.selected_b,
                origin_uv=choice.signature.origin_uv,
                grid_shape=choice.signature.cells.shape,
                budget=policy.budget, phase_anchor_sha256=anchor_sha)
            replayed = _deep_replay(result)
            if isinstance(result, plane_region_support.AlignedRegionSupport):
                produced.append(RegionSupportEvidence(
                    identity.identity_sha256, choice.signature_sha256,
                    plane_region_support.usable_mask_sha256(mask.mask),
                    frame.image_sha256, result))
            elif isinstance(result, plane_region_support.AlignedSupportRefusal):
                refused.append(replayed)
            else:
                raise ValueError("V1 producer returned a foreign type")
    except _ERRORS as error:
        return _refusal("provenance_mismatch", "region_support",
                        type(error).__name__)
    if refused:
        status = ("budget_exhausted" if all(item.reason == "budget_exhausted"
                                            for item in refused) else "refused")
        return _make(anchor, signatures, status=status,
                     reason="region_support_" + status, refusals=refused)
    return _make(anchor, signatures, status="assembled", evidence=produced,
                 anchor_sha=anchor_sha)
