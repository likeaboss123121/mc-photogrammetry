"""Derive V2/V3/M5 state output from an accepted image-derived geometry result."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

from mc_photogrammetry.pipeline import image_derived_geometry_orchestration as r1
from mc_photogrammetry.pipeline import image_derived_output_policy as policy_mod
from mc_photogrammetry.pipeline import image_derived_signatures as f1_mod
from mc_photogrammetry.reconstruction import appearance_state_proposals as proposals_mod
from mc_photogrammetry.reconstruction import plane_face_observation as face_mod
from mc_photogrammetry.reconstruction import plane_placement as placement_mod
from mc_photogrammetry.reconstruction import plane_volume as volume_mod
from mc_photogrammetry.reconstruction import state_promotion as promotion_mod

__all__ = ["IMAGE_DERIVED_OUTPUT_SCHEMA", "IMAGE_DERIVED_OUTPUT_PRODUCER",
           "ImageDerivedOutputResult", "derive_image_derived_output"]

IMAGE_DERIVED_OUTPUT_SCHEMA = "mc-photogrammetry-image-derived-output"
IMAGE_DERIVED_OUTPUT_PRODUCER = "r2-a-image-derived-output-1"
IMAGE_DERIVED_OUTPUT_TERMINAL = "geometry_unavailable"
_ERRORS = (AttributeError, TypeError, ValueError, IndexError, KeyError,
           OverflowError, ZeroDivisionError, FloatingPointError)


def _sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hex(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _deep_replay(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{field.name: getattr(value, field.name)
                              for field in dataclasses.fields(value)})
    if isinstance(value, tuple):
        return tuple(_deep_replay(item) for item in value)
    if isinstance(value, list):
        return [_deep_replay(item) for item in value]
    if isinstance(value, dict):
        return {_deep_replay(key): _deep_replay(item)
                for key, item in value.items()}
    return value


def _record_digest(value: object | None) -> str:
    if value is None:
        return ""
    for name in ("placement_sha256", "volume_sha256", "context_sha256",
                 "result_sha256", "result_digest", "promotion_sha256",
                 "refusal_sha256"):
        digest = getattr(value, name, None)
        if _hex(digest):
            return digest
    raise ValueError("output record has no canonical digest")


def _payload(value: "ImageDerivedOutputResult") -> dict:
    return {
        "schema": value.schema, "producer": value.producer,
        "terminal": value.terminal, "status": value.status,
        "reason": value.reason, "detail": value.detail,
        "geometry_sha256": value.geometry_sha256,
        "session_sha256": value.session_sha256,
        "policy_sha256": value.policy_sha256,
        "placement_sha256": _record_digest(value.placement),
        "volume_sha256": _record_digest(value.volume),
        "frame_sha256": [_record_digest(item) for item in value.frames],
        "observations_sha256": _record_digest(value.face_observations),
        "proposals_sha256": _record_digest(value.proposals),
        "promotion_sha256": _record_digest(value.promotion),
    }


@dataclass(frozen=True, slots=True)
class ImageDerivedOutputResult:
    schema: str
    producer: str
    terminal: str
    status: str
    reason: str
    detail: str
    geometry_sha256: str
    session_sha256: str
    policy_sha256: str
    geometry_result: r1.ImageDerivedGeometryResult
    policy: policy_mod.ImageDerivedOutputPolicy
    placement: object | None
    volume: object | None
    frames: tuple[face_mod.FrameRectificationContext, ...]
    face_observations: object | None
    proposals: proposals_mod.FaceStateProposalResult | None
    promotion: object | None
    result_sha256: str = ""

    def __post_init__(self) -> None:
        if (self.schema != IMAGE_DERIVED_OUTPUT_SCHEMA
                or self.producer != IMAGE_DERIVED_OUTPUT_PRODUCER
                or self.terminal != IMAGE_DERIVED_OUTPUT_TERMINAL
                or self.status not in (
                    "promoted", "review", "refused", "budget_exhausted",
                    "unavailable")
                or type(self.reason) is not str or type(self.detail) is not str
                or not all(_hex(item) for item in (
                    self.geometry_sha256, self.session_sha256,
                    self.policy_sha256))
                or not isinstance(self.geometry_result,
                                  r1.ImageDerivedGeometryResult)
                or not isinstance(self.policy,
                                  policy_mod.ImageDerivedOutputPolicy)
                or not isinstance(self.frames, tuple)):
            raise ValueError("image-derived output is malformed")
        geometry = _deep_replay(self.geometry_result)
        policy = _deep_replay(self.policy)
        records = tuple(_deep_replay(value) for value in (
            self.placement, self.volume, self.face_observations,
            self.proposals, self.promotion) if value is not None)
        frames = tuple(_deep_replay(item) for item in self.frames)
        if (geometry.result_sha256 != self.geometry_sha256
                or policy.policy_sha256 != self.policy_sha256
                or policy.to_record()
                    != policy_mod.IMAGE_DERIVED_OUTPUT_POLICY.to_record()):
            raise ValueError("image-derived output sources do not bind")
        if self.status == "unavailable":
            if (not self.reason or records or frames
                    or geometry.status == "assembled"):
                raise ValueError("unavailable output must be atomic")
        elif geometry.status != "assembled":
            raise ValueError("output requires assembled geometry")
        if self.status == "promoted":
            if (self.reason or self.detail
                    or not isinstance(self.placement,
                                      placement_mod.ResolvedFacePlacement)
                    or not isinstance(self.volume, volume_mod.PlaneVolumeResult)
                    or not frames
                    or not isinstance(self.face_observations,
                                      face_mod.PlaneFaceObservationResult)
                    or not isinstance(self.proposals,
                                      proposals_mod.FaceStateProposalResult)
                    or self.proposals.status != proposals_mod.PROPOSED
                    or not isinstance(self.promotion,
                                      promotion_mod.PromotionResult)):
                raise ValueError("promoted output is inconsistent")
        elif self.status != "unavailable" and not self.reason:
            raise ValueError("non-promoted output requires a reason")
        if isinstance(self.volume, volume_mod.PlaneVolumeResult):
            if (not isinstance(self.placement,
                               placement_mod.ResolvedFacePlacement)
                    or self.volume.placement_sha256
                        != self.placement.placement_sha256):
                raise ValueError("volume does not bind placement")
        if isinstance(self.face_observations,
                      face_mod.PlaneFaceObservationResult):
            if (not isinstance(self.volume, volume_mod.PlaneVolumeResult)
                    or self.face_observations.volume_sha256
                        != self.volume.volume_sha256
                    or self.face_observations.placement_sha256
                        != self.placement.placement_sha256
                    or len(self.face_observations.frame_bindings)
                        != len(frames)):
                raise ValueError("face observations do not bind their sources")
            context_index = {item.frame_id: item for item in frames}
            for binding in self.face_observations.frame_bindings:
                context = context_index.get(binding.get("frame_id"))
                if (context is None
                        or binding.get("image_sha256") != context.image_sha256
                        or binding.get("usable_mask_sha256")
                            != context.usable_mask_sha256
                        or binding.get("context_sha256")
                            != context.context_sha256):
                    raise ValueError("face frame bindings do not replay")
        if self.proposals is not None:
            if (not isinstance(self.face_observations,
                               face_mod.PlaneFaceObservationResult)
                    or self.proposals.observation_count
                        != len(self.face_observations.observations)
                    or self.proposals.palette_size != len(policy.palette)
                    or self.proposals.max_cost != policy.max_cost
                    or self.proposals.min_margin != policy.min_margin
                    or self.proposals.max_observations
                        != policy.max_observations
                    or self.proposals.max_candidate_evaluations
                        != policy.max_candidate_evaluations
                    or self.proposals.max_payload_bytes
                        != policy.max_proposal_payload_bytes):
                raise ValueError("proposals do not bind policy or observations")
        if isinstance(self.promotion, promotion_mod.PromotionResult):
            if (not isinstance(self.volume, volume_mod.PlaneVolumeResult)
                    or self.proposals is None
                    or self.promotion.volume_sha256 != self.volume.volume_sha256
                    or self.promotion.proposal_result_digest
                        != self.proposals.result_digest):
                raise ValueError("promotion does not bind output sources")
        digest = _sha(_payload(self))
        if self.result_sha256 not in ("", digest):
            raise ValueError("image-derived output digest mismatch")
        object.__setattr__(self, "result_sha256", digest)


def _refusal(reason: str, field: str, error: object = ""):
    detail = {"field": field}
    if error:
        detail["detail"] = str(error)
    return placement_mod._make_refusal(reason, detail)


def _make(geometry, session_sha, *, status, reason="", detail="",
          placement=None, volume=None, frames=(), observations=None,
          proposals=None, promotion=None):
    policy = policy_mod.IMAGE_DERIVED_OUTPUT_POLICY
    return ImageDerivedOutputResult(
        IMAGE_DERIVED_OUTPUT_SCHEMA, IMAGE_DERIVED_OUTPUT_PRODUCER,
        IMAGE_DERIVED_OUTPUT_TERMINAL, status, reason, detail,
        geometry.result_sha256, session_sha, policy.policy_sha256,
        geometry, policy, placement, volume, tuple(frames), observations,
        proposals, promotion)


def _failure_status(reason: str) -> str:
    return ("budget_exhausted" if reason in (
        "budget_exhausted", "dense_bounds_exceeded", proposals_mod.BUDGET)
            else "refused")


def _derive_ready(*, geometry, live_frames, policy):
    universe = geometry.universe
    supports = geometry.supports
    placed = placement_mod.resolve_face_placement(
        universe=universe, supports=supports, budget=policy.placement_budget)
    if isinstance(placed, placement_mod.PlacementRefusal):
        return _failure_status(placed.reason), "placement_" + placed.reason, \
            placed, None, (), None, None, None
    built_volume = volume_mod.build_plane_volume(
        universe=universe, supports=supports, placement=placed,
        placement_budget=policy.placement_budget, budget=policy.volume_budget)
    if isinstance(built_volume, volume_mod.VolumeRefusal):
        return _failure_status(built_volume.reason), \
            "volume_" + built_volume.reason, placed, built_volume, (), None, None, None
    records = dict(geometry.records)
    f1 = records["f1"]
    p2 = records["p2"]
    lattice = p2.lattice_result
    masks = {item.frame_id: item.mask for item in f1.usable_masks}
    contexts = []
    for frame in live_frames:
        readings = tuple(item for item in lattice.readings
                         if item.frame == frame.frame_id)
        mask = masks.get(frame.frame_id)
        if frame.image is None or mask is None or not readings:
            raise ValueError("frame context join is incomplete")
        contexts.append(face_mod.frame_context_from_readings(
            frame_id=frame.frame_id, image=frame.image,
            image_sha256=frame.image_sha256, usable_mask=mask,
            readings=readings))
    observations = face_mod.build_plane_face_observations(
        built_volume, universe=universe, supports=supports, placement=placed,
        placement_budget=policy.placement_budget,
        volume_budget=policy.volume_budget, frames=tuple(contexts),
        budget=policy.face_budget)
    if isinstance(observations, face_mod.PlaneFaceObservationRefusal):
        return _failure_status(observations.reason), \
            "face_observations_" + observations.reason, placed, built_volume, \
            tuple(contexts), observations, None, None
    proposed = proposals_mod.propose_face_states(
        observations.observations, policy.palette,
        max_cost=policy.max_cost, min_margin=policy.min_margin,
        max_observations=policy.max_observations,
        max_candidate_evaluations=policy.max_candidate_evaluations,
        max_payload_bytes=policy.max_proposal_payload_bytes)
    if proposed.status == proposals_mod.REVIEW:
        return "review", "proposals_" + str(proposed.reason), placed, \
            built_volume, tuple(contexts), observations, proposed, None
    if proposed.status != proposals_mod.PROPOSED:
        return "budget_exhausted", "proposals_" + str(proposed.reason), placed, \
            built_volume, tuple(contexts), observations, proposed, None
    promoted = promotion_mod.promote_plane_states(
        built_volume, proposed, budget=policy.promotion_budget,
        allow_omitted_unknown=policy.allow_omitted_unknown)
    if isinstance(promoted, promotion_mod.PromotionRefusal):
        status = _failure_status(promoted.reason)
        return status, "promotion_" + promoted.reason, placed, built_volume, \
            tuple(contexts), observations, proposed, promoted
    if not isinstance(promoted, promotion_mod.PromotionResult):
        raise ValueError("promotion producer returned a foreign record")
    return "promoted", "", placed, built_volume, tuple(contexts), \
        observations, proposed, promoted


def derive_image_derived_output(
    *, geometry_result: object, session: object,
) -> ImageDerivedOutputResult | placement_mod.PlacementRefusal:
    """Derive exact state output from one replayed R1 result and live pixels."""
    if not isinstance(geometry_result, r1.ImageDerivedGeometryResult):
        return _refusal("invalid_input", "geometry_result")
    try:
        geometry = _deep_replay(geometry_result)
    except _ERRORS as error:
        return _refusal("provenance_mismatch", "geometry_result",
                        type(error).__name__)
    replayed_session = f1_mod._replay_session(session)
    if isinstance(replayed_session, placement_mod.PlacementRefusal):
        return replayed_session
    checked_session, live_frames = replayed_session
    if checked_session.session_sha256 != geometry.session_sha256:
        return _refusal("provenance_mismatch", "session.geometry")
    if geometry.status != "assembled":
        return _make(geometry, checked_session.session_sha256,
                     status="unavailable", reason="geometry_" + geometry.status,
                     detail=geometry.reason)
    policy = policy_mod.IMAGE_DERIVED_OUTPUT_POLICY
    try:
        values = _derive_ready(geometry=geometry, live_frames=live_frames,
                               policy=policy)
        status, reason, placed, built_volume, frames, observations, proposed, promoted = values
        return _make(geometry, checked_session.session_sha256, status=status,
                     reason=reason, placement=placed, volume=built_volume,
                     frames=frames, observations=observations,
                     proposals=proposed, promotion=promoted)
    except _ERRORS as error:
        return _refusal("provenance_mismatch", "output_derivation",
                        type(error).__name__)
