"""Frozen production policy for image-derived V2/V3/M5 output."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from mc_photogrammetry.blocks.appearance import DEFAULT_PALETTE, PaletteColor
from mc_photogrammetry.reconstruction import appearance_state_proposals as proposals
from mc_photogrammetry.reconstruction import plane_face_observation as face
from mc_photogrammetry.reconstruction import plane_placement as placement
from mc_photogrammetry.reconstruction import plane_volume as volume
from mc_photogrammetry.reconstruction import state_promotion as promotion

__all__ = ["IMAGE_DERIVED_OUTPUT_POLICY_SCHEMA",
           "IMAGE_DERIVED_OUTPUT_POLICY_PRODUCER",
           "IMAGE_DERIVED_OUTPUT_POLICY_VERSION", "ImageDerivedOutputPolicy",
           "IMAGE_DERIVED_OUTPUT_POLICY"]

IMAGE_DERIVED_OUTPUT_POLICY_SCHEMA = \
    "mc-photogrammetry-image-derived-output-policy"
IMAGE_DERIVED_OUTPUT_POLICY_PRODUCER = "r2-p-image-derived-output-policy-1"
IMAGE_DERIVED_OUTPUT_POLICY_VERSION = 1


def _sha(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hex(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _caps(record: dict) -> dict:
    return {key: value for key, value in record.items()
            if key != "budget_sha256"}


def _ordered_palette(palette: tuple[PaletteColor, ...]) -> list[dict]:
    return [proposals._palette_payload(item) for item in palette]


def _payload(*, schema, producer, version, placement_budget, volume_budget,
             face_budget, palette, max_cost, min_margin, max_observations,
             max_candidate_evaluations, max_proposal_payload_bytes,
             promotion_budget, allow_omitted_unknown, palette_sha256) -> dict:
    return {
        "schema": schema, "producer": producer, "version": version,
        "placement_budget": placement_budget.to_record(),
        "volume_budget": volume_budget.to_record(),
        "face_budget": face_budget.to_record(),
        "palette": _ordered_palette(palette),
        "palette_sha256": palette_sha256,
        "max_cost": max_cost, "min_margin": min_margin,
        "max_observations": max_observations,
        "max_candidate_evaluations": max_candidate_evaluations,
        "max_proposal_payload_bytes": max_proposal_payload_bytes,
        "promotion_budget": promotion_budget.to_record(),
        "allow_omitted_unknown": allow_omitted_unknown,
    }


@dataclass(frozen=True, slots=True)
class ImageDerivedOutputPolicy:
    schema: str
    producer: str
    version: int
    placement_budget: placement.PlacementBudget
    volume_budget: volume.PlaneVolumeBudget
    face_budget: face.PlaneFaceObservationBudget
    palette: tuple[PaletteColor, ...]
    max_cost: float
    min_margin: float
    max_observations: int
    max_candidate_evaluations: int
    max_proposal_payload_bytes: int
    promotion_budget: promotion.PromotionBudget
    allow_omitted_unknown: bool
    palette_sha256: str
    policy_sha256: str

    def __post_init__(self) -> None:
        if self.schema != IMAGE_DERIVED_OUTPUT_POLICY_SCHEMA:
            raise ValueError("unsupported output policy schema")
        if self.producer != IMAGE_DERIVED_OUTPUT_POLICY_PRODUCER:
            raise ValueError("unsupported output policy producer")
        if (type(self.version) is not int
                or self.version != IMAGE_DERIVED_OUTPUT_POLICY_VERSION):
            raise ValueError("unsupported output policy version")
        try:
            placement_record = self.placement_budget.to_record()
            volume_record = self.volume_budget.to_record()
            face_record = self.face_budget.to_record()
            promotion_record = self.promotion_budget.to_record()
            replayed_placement = placement.make_placement_budget(
                **_caps(placement_record))
            replayed_volume = volume.make_volume_budget(**_caps(volume_record))
            replayed_face = face.make_face_observation_budget(
                **_caps(face_record))
            replayed_promotion = promotion.make_promotion_budget(
                **_caps(promotion_record))
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("output policy budgets do not replay") from error
        if (replayed_placement.to_record() != placement_record
                or replayed_volume.to_record() != volume_record
                or replayed_face.to_record() != face_record
                or replayed_promotion.to_record() != promotion_record):
            raise ValueError("output policy budgets do not replay exactly")
        try:
            validated_palette = proposals._validate_palette(self.palette)
        except (TypeError, ValueError, proposals.AppearanceStateProposalError) as error:
            raise ValueError("output policy palette does not replay") from error
        if validated_palette != DEFAULT_PALETTE:
            raise ValueError("output policy requires the exact default palette")
        palette_digest = proposals._palette_digest(validated_palette)
        if not _hex(self.palette_sha256) or self.palette_sha256 != palette_digest:
            raise ValueError("output policy palette digest mismatch")
        if (type(self.max_cost) is not float or self.max_cost != 1e9
                or type(self.min_margin) is not float or self.min_margin != 0.0
                or type(self.max_observations) is not int
                or self.max_observations != 1000
                or type(self.max_candidate_evaluations) is not int
                or self.max_candidate_evaluations != 100_000
                or type(self.max_proposal_payload_bytes) is not int
                or self.max_proposal_payload_bytes != 10_000_000
                or self.allow_omitted_unknown is not False):
            raise ValueError("output policy values are not the built-in v1 values")
        expected = _sha(_payload(
            schema=self.schema, producer=self.producer, version=self.version,
            placement_budget=replayed_placement, volume_budget=replayed_volume,
            face_budget=replayed_face, palette=validated_palette,
            max_cost=self.max_cost, min_margin=self.min_margin,
            max_observations=self.max_observations,
            max_candidate_evaluations=self.max_candidate_evaluations,
            max_proposal_payload_bytes=self.max_proposal_payload_bytes,
            promotion_budget=replayed_promotion,
            allow_omitted_unknown=self.allow_omitted_unknown,
            palette_sha256=palette_digest))
        if not _hex(self.policy_sha256) or self.policy_sha256 != expected:
            raise ValueError("output policy digest mismatch")

    def to_record(self) -> dict:
        record = _payload(
            schema=self.schema, producer=self.producer, version=self.version,
            placement_budget=self.placement_budget,
            volume_budget=self.volume_budget, face_budget=self.face_budget,
            palette=self.palette, max_cost=self.max_cost,
            min_margin=self.min_margin, max_observations=self.max_observations,
            max_candidate_evaluations=self.max_candidate_evaluations,
            max_proposal_payload_bytes=self.max_proposal_payload_bytes,
            promotion_budget=self.promotion_budget,
            allow_omitted_unknown=self.allow_omitted_unknown,
            palette_sha256=self.palette_sha256)
        record["policy_sha256"] = self.policy_sha256
        return record


def _builtin() -> ImageDerivedOutputPolicy:
    values = dict(
        schema=IMAGE_DERIVED_OUTPUT_POLICY_SCHEMA,
        producer=IMAGE_DERIVED_OUTPUT_POLICY_PRODUCER,
        version=IMAGE_DERIVED_OUTPUT_POLICY_VERSION,
        placement_budget=placement.make_placement_budget(
            max_faces=256, max_supports_checked=64, max_support_cells=4096,
            max_frame_axis_lookups=4096, max_integer_checks=4096,
            max_anchors=65536),
        volume_budget=volume.make_volume_budget(
            max_faces=10_000, max_support_cells=10_000_000,
            max_contributions=10_000_000, max_coordinates=10_000_000,
            max_conflicts=100_000, max_omitted_unknown=1,
            max_dense_bounds_cells=10_000_000),
        face_budget=face.make_face_observation_budget(
            max_claims=10_000_000, max_frames=32,
            max_rectified_pixels=100_000_000, max_payload_bytes=100_000_000,
            max_replay_checks=1_000_000),
        palette=DEFAULT_PALETTE, max_cost=1e9, min_margin=0.0,
        max_observations=1000, max_candidate_evaluations=100_000,
        max_proposal_payload_bytes=10_000_000,
        promotion_budget=promotion.make_promotion_budget(
            max_cells=10_000_000, max_proposals=10_000_000,
            max_payload_bytes=100_000_000),
        allow_omitted_unknown=False,
        palette_sha256=proposals._palette_digest(DEFAULT_PALETTE))
    return ImageDerivedOutputPolicy(
        **values, policy_sha256=_sha(_payload(**values)))


IMAGE_DERIVED_OUTPUT_POLICY = _builtin()
