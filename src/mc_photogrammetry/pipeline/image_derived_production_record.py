"""Durable schema-3 image-derived production record codec.

This module owns the canonical, digest-bound *terminal envelope* that a schema-3
``.mcscan`` project carries alongside a schema-2 session record and a top-level
export volume.  It is the only seam that turns a live
:class:`~mc_photogrammetry.pipeline.image_derived_production.ImageDerivedProductionResult`
(or a typed placement refusal) into a pixel-free, JSON-native record, and the
only seam that turns such a record back into a replay-valid
:class:`~mc_photogrammetry.reconstruction.state_promotion.PromotionResult`.

Contract
--------

* **Compact and pixel-free.**  The envelope carries one schema name, one producer
  identity, the terminal ``stage``/``status``/``reason``/``detail``, the accepted
  stage digests, the schema-2 session digest, a volume digest, an optional
  placement-refusal digest, and - only for an exact-state ``promoted`` terminal -
  the full serialized ``PromotionResult``.  It carries no pixels, no camera
  manifest, no F3 reference, no manual coordinate/direction/bound, and no runtime
  object.  The whole canonical record is bounded by
  :data:`PRODUCTION_RECORD_MAX_BYTES`.
* **Promoted iff exact.**  ``status == "promoted"`` requires ``stage == "output"``,
  an empty reason/detail, and a reconstructed promotion that is literally
  ``export_ready``, literally ``exact_state_eligible``, and has
  ``omitted_unknown == 0``.  Every other status must carry *no* promotion and a
  volume digest that is exactly the digest of the empty volume, so a refusal,
  review, omission, exhausted budget or unavailable geometry can never persist
  fabricated blocks.
* **Canonical stage prefix.**  A terminal at stage *S* carries the digest of every
  completed prior stage and no later-stage digest.  A normal result at *S* also
  carries *S*'s digest and no refusal digest; a direct
  ``PlacementRefusal`` - identified by the presence of ``refusal_sha256``, not by
  the status string - carries the refusal digest and omits *S*'s digest, while the
  entry digest (the base binding) is always present and the entry stage can never
  be a direct refusal.
* **Strict replay.**  Loading a record rebuilds every nested promoted block
  through :meth:`PromotedBlock.build`, rebuilds the enclosing result through
  :meth:`PromotionResult.build`, recomputes every digest, and requires the
  rebuilt record to serialize back to exactly the provided record.  A stale,
  resealed, truncated or unknown field is a typed
  :class:`ProductionEnvelopeError` rather than a partially trusted record.
* **Volume binding.**  ``volume_sha256`` is the canonical digest of the exact
  block/state/confidence/frame evidence set; the persistence boundary recomputes
  it from the stored volume and additionally requires the stored volume to equal
  ``promotion.to_block_volume()`` block-for-block.  An exact-state project can
  therefore be exported only when its stored volume replays its promotion.

The module performs no image, camera, manifest, exporter or project-store work;
the campaign/CLI layer owns the actual no-clobber publication.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mc_photogrammetry.domain.models import BlockVolume
from mc_photogrammetry.reconstruction import plane_placement
from mc_photogrammetry.reconstruction import state_promotion

__all__ = [
    "PRODUCTION_RECORD_SCHEMA",
    "PRODUCTION_RECORD_PRODUCER",
    "PRODUCTION_RECORD_MAX_BYTES",
    "PRODUCTION_RECORD_STAGES",
    "PRODUCTION_RECORD_STATUSES",
    "ProductionEnvelopeError",
    "ProductionEnvelope",
    "build_production_envelope",
    "production_envelope_from_record",
    "canonicalise_production_record",
    "production_volume",
    "refusal_detail",
    "target_refusal",
    "storage_refusal",
    "checkpoint_path",
    "cleanup_checkpoint",
]

PRODUCTION_RECORD_SCHEMA = "mc-photogrammetry-image-derived-production-record-3"
"""Explicit schema name for the schema-3 terminal envelope."""

PRODUCTION_RECORD_PRODUCER = "ds-r9-durable-production-record-2"
_HISTORICAL_PRODUCTION_RECORD_PRODUCER = "ds-r9-durable-production-record-1"
"""Producer identity carried by every schema-3 terminal envelope."""

PRODUCTION_RECORD_MAX_BYTES = 128 * 1024 * 1024
"""Hard bound on one canonical serialized envelope, in UTF-8 bytes."""

PRODUCTION_RECORD_STAGES: tuple[str, ...] = ("entry", "lattice", "geometry", "output")
"""The closed terminal-stage enum, in canonical order."""

PRODUCTION_RECORD_STATUSES: tuple[str, ...] = (
    "promoted",
    "review",
    "refused",
    "geometry_unavailable",
    "budget_exhausted",
    "unavailable",
)
"""The closed terminal-status enum; only ``promoted`` may carry a promotion."""

_HISTORICAL_RECORD_KEYS = frozenset({
    "schema", "producer", "stage", "status", "reason", "detail",
    "entry_sha256", "lattice_sha256", "geometry_sha256", "output_sha256",
    "session_sha256", "volume_sha256", "promotion", "refusal_sha256",
    "record_sha256",
})
_RECORD_KEYS = _HISTORICAL_RECORD_KEYS | {"joint_lineage"}

# Complete graph/selection identity pairs; versions cannot be mixed on replay.
_JOINT_LINEAGE_IDENTITIES = (
    ("mc-photogrammetry-image-derived-graph-v3", "m9-g-image-derived-graph-7",
     "mc-photogrammetry-endpoint-selection-v1", "m9-g-endpoint-selection-1"),
    ("mc-photogrammetry-image-derived-graph-v4", "m9-g-image-derived-graph-8",
     "mc-photogrammetry-endpoint-selection-v2", "m9-g-endpoint-selection-2"),
    ("mc-photogrammetry-image-derived-graph-v5", "m9-g-image-derived-graph-9",
     "mc-photogrammetry-endpoint-selection-v3", "m9-g-endpoint-selection-3"),
    ("mc-photogrammetry-image-derived-graph-v6", "m9-g-image-derived-graph-10",
     "mc-photogrammetry-endpoint-selection-v4", "m9-g-endpoint-selection-4"),
)


def _validate_joint_lineage(value):
    if type(value) is not tuple or len(value) not in (0, 6):
        raise ProductionEnvelopeError("joint lineage is malformed")
    if value and ((value[0], value[1], value[3], value[4]) not in _JOINT_LINEAGE_IDENTITIES
            or not _hex64(value[2]) or not _hex64(value[5])):
        raise ProductionEnvelopeError("joint lineage identity/digest is unsupported")

_STAGE_DIGEST = {
    "entry": "entry_sha256",
    "lattice": "lattice_sha256",
    "geometry": "geometry_sha256",
    "output": "output_sha256",
}

#: Digests of the stages that must already be complete at each terminal stage.
_PRIOR_STAGE_DIGESTS: dict[str, tuple[str, ...]] = {
    "entry": (),
    "lattice": ("entry_sha256",),
    "geometry": ("entry_sha256", "lattice_sha256"),
    "output": ("entry_sha256", "lattice_sha256", "geometry_sha256"),
}

#: Digests of the stages that must not exist yet at each terminal stage.
_LATER_STAGE_DIGESTS: dict[str, tuple[str, ...]] = {
    "entry": ("lattice_sha256", "geometry_sha256", "output_sha256"),
    "lattice": ("geometry_sha256", "output_sha256"),
    "geometry": ("output_sha256",),
    "output": (),
}

_ERRORS = (
    AttributeError,
    TypeError,
    ValueError,
    IndexError,
    KeyError,
    OverflowError,
    ZeroDivisionError,
)


class ProductionEnvelopeError(ValueError):
    """Raised when a schema-3 terminal envelope cannot be trusted or replayed."""


# ---------------------------------------------------------------------------
# Canonical JSON and small exact helpers
# ---------------------------------------------------------------------------

def _json_value(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON values must be finite")
        return float(value)
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    raise ValueError("value is not a deterministic JSON primitive")


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _hex64(value: Any) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _hex64_or_empty(value: Any) -> bool:
    return value == "" or _hex64(value)


def _volume_digest(volume: BlockVolume) -> str:
    """Canonical digest of the exact stored block/state/confidence/frame evidence."""
    return _sha256(volume.to_dict())


# ---------------------------------------------------------------------------
# Terminal envelope
# ---------------------------------------------------------------------------

def _record_payload(value: "ProductionEnvelope") -> dict:
    payload = {
        "schema": value.schema,
        "producer": value.producer,
        "stage": value.stage,
        "status": value.status,
        "reason": value.reason,
        "detail": value.detail,
        "entry_sha256": value.entry_sha256,
        "lattice_sha256": value.lattice_sha256,
        "geometry_sha256": value.geometry_sha256,
        "output_sha256": value.output_sha256,
        "session_sha256": value.session_sha256,
        "volume_sha256": value.volume_sha256,
        "promotion": (None if value.promotion is None
                      else value.promotion.to_record()),
        "refusal_sha256": value.refusal_sha256,
    }
    if value.producer == PRODUCTION_RECORD_PRODUCER:
        payload["joint_lineage"] = list(value.joint_lineage)
    return payload


@dataclass(frozen=True, slots=True)
class ProductionEnvelope:
    """One immutable, digest-bound schema-3 terminal envelope.

    The record binds the exact session digest, the accepted stage digests, the
    terminal stage/status/reason/detail, the canonical volume digest and (only for
    an exact-state promotion) the full serialized ``PromotionResult``.  It is
    constructed and re-validated in :meth:`__post_init__`, so a caller cannot hold
    a malformed or over-sized envelope.
    """

    schema: str
    producer: str
    stage: str
    status: str
    reason: str
    detail: str
    entry_sha256: str
    lattice_sha256: str
    geometry_sha256: str
    output_sha256: str
    session_sha256: str
    volume_sha256: str
    promotion: state_promotion.PromotionResult | None
    refusal_sha256: str
    joint_lineage: tuple = ()
    record_sha256: str = ""

    def __post_init__(self) -> None:
        if (self.schema != PRODUCTION_RECORD_SCHEMA
                or self.producer not in (PRODUCTION_RECORD_PRODUCER, _HISTORICAL_PRODUCTION_RECORD_PRODUCER)):
            raise ProductionEnvelopeError("production record version is unsupported")
        _validate_joint_lineage(self.joint_lineage)
        if self.producer == _HISTORICAL_PRODUCTION_RECORD_PRODUCER and self.joint_lineage:
            raise ProductionEnvelopeError("historical envelope cannot claim joint lineage")
        if self.joint_lineage and self.stage not in ("geometry", "output"):
            raise ProductionEnvelopeError("joint lineage precedes geometry")
        if (self.producer == PRODUCTION_RECORD_PRODUCER and self.stage == "output"
                and not self.joint_lineage):
            raise ProductionEnvelopeError("production output lacks joint lineage")
        if self.stage not in PRODUCTION_RECORD_STAGES:
            raise ProductionEnvelopeError("production record stage is not in the closed enum")
        if self.status not in PRODUCTION_RECORD_STATUSES:
            raise ProductionEnvelopeError("production record status is not in the closed enum")
        if type(self.reason) is not str or type(self.detail) is not str:
            raise ProductionEnvelopeError("production record reason/detail must be strings")
        if not _hex64(self.entry_sha256):
            raise ProductionEnvelopeError("production record entry digest is malformed")
        if not (_hex64_or_empty(self.lattice_sha256)
                and _hex64_or_empty(self.geometry_sha256)
                and _hex64_or_empty(self.output_sha256)
                and _hex64_or_empty(self.refusal_sha256)):
            raise ProductionEnvelopeError("production record stage digest is malformed")
        if not _hex64(self.session_sha256):
            raise ProductionEnvelopeError("production record session digest is malformed")
        if not _hex64(self.volume_sha256):
            raise ProductionEnvelopeError("production record volume digest is malformed")
        # Canonical stage prefix: every completed prior stage is bound, no later
        # stage exists yet, and the current stage is either carried (a normal
        # result) or explicitly omitted (a direct placement refusal).
        for name in _PRIOR_STAGE_DIGESTS[self.stage]:
            if not _hex64(getattr(self, name)):
                raise ProductionEnvelopeError(
                    "production record is missing a completed stage digest")
        for name in _LATER_STAGE_DIGESTS[self.stage]:
            if getattr(self, name):
                raise ProductionEnvelopeError(
                    "production record carries a later-stage digest")
        current = _STAGE_DIGEST[self.stage]
        if self.refusal_sha256:
            # A refusal digest, not the status string, identifies a direct
            # PlacementRefusal: an ordinary stage result may itself carry the
            # status "refused" while still holding its completed stage digest.
            if self.status != "refused":
                raise ProductionEnvelopeError(
                    "a direct refusal must carry the refused status")
            if self.stage == "entry":
                raise ProductionEnvelopeError(
                    "the entry stage cannot carry a direct refusal")
            if getattr(self, current):
                raise ProductionEnvelopeError(
                    "a direct refusal must omit its current stage digest")
        elif not _hex64(getattr(self, current)):
            raise ProductionEnvelopeError(
                "production record is missing its current stage digest")
        if self.status == "promoted":
            if (self.stage != "output" or self.reason or self.detail
                    or self.refusal_sha256
                    or not isinstance(self.promotion,
                                      state_promotion.PromotionResult)):
                raise ProductionEnvelopeError("promoted production record is malformed")
            promotion = self.promotion
            if (promotion.export_ready is not True
                    or promotion.exact_state_eligible is not True
                    or promotion.omitted_unknown != 0
                    or not promotion.cells):
                raise ProductionEnvelopeError(
                    "promoted production record is not exact-state eligible")
            if self.volume_sha256 != _volume_digest(promotion.to_block_volume()):
                raise ProductionEnvelopeError(
                    "promoted production record does not bind its volume")
        else:
            if self.promotion is not None:
                raise ProductionEnvelopeError(
                    "a non-promoted production record cannot carry a promotion")
            if not self.reason:
                raise ProductionEnvelopeError(
                    "a non-promoted production record needs a typed reason")
            if self.volume_sha256 != _volume_digest(BlockVolume()):
                raise ProductionEnvelopeError(
                    "a non-promoted production record must not claim a volume")
        payload = _record_payload(self)
        encoded = _canonical_json(payload)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        if self.record_sha256 not in ("", digest):
            raise ProductionEnvelopeError("production record digest mismatch")
        object.__setattr__(self, "record_sha256", digest)
        # The bound applies to the complete serialized record, including the
        # 64-character ``record_sha256`` member, not only the pre-digest payload.
        if len(_canonical_json(self.to_record()).encode("utf-8")) \
                > PRODUCTION_RECORD_MAX_BYTES:
            raise ProductionEnvelopeError(
                "production record exceeds the serialized size bound")

    def to_record(self) -> dict:
        record = _record_payload(self)
        record["record_sha256"] = self.record_sha256
        return record


def build_production_envelope(
    *,
    stage: str,
    status: str,
    reason: str,
    detail: str = "",
    entry_sha256: str,
    session_sha256: str,
    lattice_sha256: str = "",
    geometry_sha256: str = "",
    output_sha256: str = "",
    promotion: state_promotion.PromotionResult | None = None,
    refusal_sha256: str = "",
    volume: BlockVolume | None = None,
    joint_lineage: tuple = (),
) -> ProductionEnvelope:
    """Build the canonical terminal envelope for one image-derived outcome."""
    resolved = volume
    if resolved is None:
        resolved = (BlockVolume() if promotion is None
                    else promotion.to_block_volume())
    return ProductionEnvelope(
        schema=PRODUCTION_RECORD_SCHEMA,
        producer=PRODUCTION_RECORD_PRODUCER,
        stage=stage,
        status=status,
        reason=reason,
        detail=detail,
        entry_sha256=entry_sha256,
        lattice_sha256=lattice_sha256,
        geometry_sha256=geometry_sha256,
        output_sha256=output_sha256,
        session_sha256=session_sha256,
        volume_sha256=_volume_digest(resolved),
        promotion=promotion,
        refusal_sha256=refusal_sha256,
        joint_lineage=joint_lineage,
    )


def production_envelope_from_record(record: Mapping[str, Any]) -> ProductionEnvelope:
    """Rebuild one serialized terminal envelope and replay every bound digest.

    The exact record key set is required (a missing or unknown key is refused),
    every nested promotion is rebuilt through
    :func:`state_promotion.promotion_result_from_record`, and the rebuilt envelope
    must serialize back to exactly the provided record.
    """
    if not isinstance(record, Mapping):
        raise ProductionEnvelopeError("production record must be a mapping")
    historical = record.get("producer") == _HISTORICAL_PRODUCTION_RECORD_PRODUCER
    if set(record) != (_HISTORICAL_RECORD_KEYS if historical else _RECORD_KEYS):
        raise ProductionEnvelopeError(
            "production record key set is not canonical")
    if not historical and (type(record["joint_lineage"]) is not list
            or len(record["joint_lineage"]) not in (0, 6)):
        raise ProductionEnvelopeError("joint lineage container/count is malformed")
    try:
        promotion_record = record["promotion"]
        promotion = (None if promotion_record is None else
                     state_promotion.promotion_result_from_record(promotion_record))
        envelope = ProductionEnvelope(
            schema=record["schema"],
            producer=record["producer"],
            stage=record["stage"],
            status=record["status"],
            reason=record["reason"],
            detail=record["detail"],
            entry_sha256=record["entry_sha256"],
            lattice_sha256=record["lattice_sha256"],
            geometry_sha256=record["geometry_sha256"],
            output_sha256=record["output_sha256"],
            session_sha256=record["session_sha256"],
            volume_sha256=record["volume_sha256"],
            promotion=promotion,
            refusal_sha256=record["refusal_sha256"],
            joint_lineage=() if historical else tuple(record["joint_lineage"]),
            record_sha256=record["record_sha256"],
        )
    except ProductionEnvelopeError:
        raise
    except _ERRORS as error:
        raise ProductionEnvelopeError(
            "production record does not rebuild") from error
    if _canonical_json(envelope.to_record()) != _canonical_json(dict(record)):
        raise ProductionEnvelopeError("production record does not replay exactly")
    return envelope


def production_volume(envelope: ProductionEnvelope) -> BlockVolume:
    """The exact block evidence a persisted terminal envelope may export.

    A promoted envelope returns its promotion's fresh volume; every other
    terminal returns an empty volume (there is nothing exact to export).
    """
    if envelope.promotion is None:
        return BlockVolume()
    return envelope.promotion.to_block_volume()


def canonicalise_production_record(
    record: Mapping[str, Any],
    *,
    session_sha256: str,
    volume: BlockVolume,
) -> dict:
    """Validate one envelope against the session and stored volume, then canonicalize.

    This is the persistence-boundary entry point: it proves the envelope replays,
    that it binds the exact schema-2 session digest, that ``volume_sha256`` is the
    digest of the *stored* volume, and that the stored volume is exactly the
    promotion's fresh volume (or exactly empty for a non-promoted terminal).
    """
    envelope = production_envelope_from_record(record)
    if not _hex64(session_sha256):
        raise ProductionEnvelopeError("stored session digest is malformed")
    if envelope.session_sha256 != session_sha256:
        raise ProductionEnvelopeError(
            "production record does not bind the stored session")
    if envelope.volume_sha256 != _volume_digest(volume):
        raise ProductionEnvelopeError(
            "production record does not bind the stored volume")
    expected = production_volume(envelope)
    if envelope.promotion is not None:
        if volume.to_dict() != expected.to_dict():
            raise ProductionEnvelopeError(
                "stored volume is not the promoted exact-state volume")
    elif volume.blocks:
        raise ProductionEnvelopeError(
            "a non-promoted production record cannot carry stored blocks")
    return envelope.to_record()


# ---------------------------------------------------------------------------
# Durable production helpers (no-clobber staging and typed refusals)
# ---------------------------------------------------------------------------

_CHECKPOINT_PREFIX = "mc-photogrammetry-entry-checkpoint-"


def target_refusal(target: Path) -> plane_placement.PlacementRefusal | None:
    """A typed refusal when the durable project target is occupied, else ``None``.

    A pre-existing destination is never overwritten: the caller refuses before any
    image discovery or decode work, so the prior bytes are preserved untouched.
    The occupied-entry test is ``os.path.lexists`` (not ``Path.exists``) so a
    dangling symbolic link still refuses entry work instead of being followed,
    replaced or removed.
    """
    if os.path.lexists(target):
        return plane_placement._make_refusal("invalid_input", {
            "field": "project_path", "reason": "target_exists",
            "name": target.name})
    return None


def storage_refusal(target: Path) -> plane_placement.PlacementRefusal:
    """A typed refusal for a staging/checkpoint/publication failure."""
    return plane_placement._make_refusal("invalid_input", {
        "field": "project_path", "reason": "publish_failed",
        "name": target.name})


def refusal_detail(refusal: plane_placement.PlacementRefusal) -> str:
    """A compact, canonical detail token for one typed placement refusal."""
    return refusal.detail_sha256


def checkpoint_path(target: Path) -> Path:
    """A private staging path for the schema-2 entry checkpoint.

    The checkpoint is written to a fresh temporary directory and removed after the
    durable schema-3 project has been published, so the requested project path is
    never touched until the final no-clobber publication.
    """
    directory = Path(tempfile.mkdtemp(prefix=_CHECKPOINT_PREFIX))
    return directory / target.name


def cleanup_checkpoint(checkpoint: Path) -> None:
    """Best-effort removal of a staged checkpoint and its temporary directory."""
    candidates = (checkpoint, checkpoint.with_suffix(checkpoint.suffix + ".tmp"))
    for candidate in candidates:
        try:
            candidate.unlink()
        except OSError:
            pass
    if not checkpoint.parent.name.startswith(_CHECKPOINT_PREFIX):
        return
    try:
        checkpoint.parent.rmdir()
    except OSError:
        pass
