"""Typed, evaluation-only screenshot ingress provenance.

One :class:`FrameIngressProvenanceV1` binds the encoded-file SHA-256 and byte
length captured from the **single** shared encoded-byte read that feeds both
discovery (content hash, byte length, de-duplication) and the decode, to the
decoded shape/dtype/pixel digest, the reversible
:class:`~mc_photogrammetry.pipeline.contracts.FrameTransform`, and the decoder
provenance. It never re-hashes or re-opens ``source_path``.

This module has no evaluator, camera, mask, Minecraft or JAR dependency; it is
imported by the session record, the signature producer and the candidate-support
record without creating an import cycle.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from mc_photogrammetry.pipeline.contracts import FrameTransform

__all__ = [
    "INGRESS_PROVENANCE_SCHEMA",
    "INGRESS_DECODER_PROVENANCE",
    "INGRESS_PROVENANCE_UNAVAILABLE",
    "BYTE_DIGEST_MISMATCH",
    "FRAME_TRANSFORM_UNAVAILABLE",
    "IngressProvenanceError",
    "FrameIngressProvenanceV1",
    "build_frame_ingress_provenance",
]

INGRESS_PROVENANCE_SCHEMA = "mc-photogrammetry-frame-ingress-provenance-v1"
INGRESS_DECODER_PROVENANCE = "input.image.load_image:exif_transpose+convert('RGB')"

INGRESS_PROVENANCE_UNAVAILABLE = "ingress_provenance_unavailable"
BYTE_DIGEST_MISMATCH = "byte_digest_mismatch"
FRAME_TRANSFORM_UNAVAILABLE = "frame_transform_unavailable"


class IngressProvenanceError(ValueError):
    """A typed, bounded ingress-pin refusal."""

    def __init__(self, reason: str, field: str = "", detail: str = "") -> None:
        self.reason = reason
        self.field = field
        self.detail = detail
        super().__init__(reason if not detail else f"{reason}: {detail}")


def _hex64(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _canonical_int(value: object, reason: str, field: str) -> int:
    """Return ``value`` only when it is a canonical ``int``.

    ``from_record`` must reject noncanonical JSON integers (``bool``, ``str``,
    ``float``) instead of silently coercing them with ``int(...)``; ``bool`` in
    particular is an ``int`` subclass and would otherwise pin ``true`` to ``1``.
    """
    if type(value) is not int:
        raise IngressProvenanceError(reason, field)
    return value


def _shape(value: object) -> tuple[int, ...]:
    if (not isinstance(value, (tuple, list)) or len(value) not in (2, 3)
            or any(type(item) is not int or item <= 0 for item in value)):
        raise IngressProvenanceError(INGRESS_PROVENANCE_UNAVAILABLE, "decoded_shape")
    shape = tuple(int(item) for item in value)
    if len(shape) == 3 and shape[2] != 3:
        raise IngressProvenanceError(INGRESS_PROVENANCE_UNAVAILABLE, "decoded_shape")
    return shape


class FrameIngressProvenanceV1:
    """One frame's screenshot ingress pin, all from the single shared read.

    Instances are immutable: every field is pinned at construction and both
    ``__setattr__`` and ``__delattr__`` refuse, so the record digest that
    ``__hash__`` derives from those fields cannot drift after a candidate-support
    join has bound to this pin.
    """

    __slots__ = ("frame_id", "source_label", "byte_sha256", "byte_length",
                 "decoded_shape", "decoded_dtype", "decoded_pixel_sha256",
                 "transform", "schema", "decoder")

    def __init__(self, *, frame_id: str, source_label: str, byte_sha256: str,
                 byte_length: int, decoded_shape: tuple[int, ...],
                 decoded_dtype: str, decoded_pixel_sha256: str,
                 transform: FrameTransform,
                 schema: str = INGRESS_PROVENANCE_SCHEMA,
                 decoder: str = INGRESS_DECODER_PROVENANCE) -> None:
        if schema != INGRESS_PROVENANCE_SCHEMA or decoder != INGRESS_DECODER_PROVENANCE:
            raise IngressProvenanceError(INGRESS_PROVENANCE_UNAVAILABLE, "schema")
        if type(frame_id) is not str or not frame_id:
            raise IngressProvenanceError(INGRESS_PROVENANCE_UNAVAILABLE, "frame_id")
        if type(source_label) is not str or not source_label:
            raise IngressProvenanceError(INGRESS_PROVENANCE_UNAVAILABLE, "source_label")
        if not _hex64(byte_sha256):
            raise IngressProvenanceError(BYTE_DIGEST_MISMATCH, "byte_sha256")
        if not _positive_int(byte_length):
            raise IngressProvenanceError(BYTE_DIGEST_MISMATCH, "byte_length")
        shape = _shape(decoded_shape)
        if decoded_dtype != "|u1":
            raise IngressProvenanceError(INGRESS_PROVENANCE_UNAVAILABLE, "decoded_dtype")
        if not _hex64(decoded_pixel_sha256):
            raise IngressProvenanceError(
                INGRESS_PROVENANCE_UNAVAILABLE, "decoded_pixel_sha256")
        if not isinstance(transform, FrameTransform):
            raise IngressProvenanceError(FRAME_TRANSFORM_UNAVAILABLE, "transform")
        if transform.resolved_output_size != (shape[1], shape[0]):
            raise IngressProvenanceError(FRAME_TRANSFORM_UNAVAILABLE, "transform.output_size")
        for name, value in (
                ("frame_id", frame_id),
                ("source_label", source_label),
                ("byte_sha256", byte_sha256),
                ("byte_length", int(byte_length)),
                ("decoded_shape", shape),
                ("decoded_dtype", decoded_dtype),
                ("decoded_pixel_sha256", decoded_pixel_sha256),
                ("transform", transform),
                ("schema", schema),
                ("decoder", decoder)):
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("FrameIngressProvenanceV1 is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("FrameIngressProvenanceV1 is immutable")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FrameIngressProvenanceV1):
            return NotImplemented
        return self.to_record() == other.to_record()

    def __hash__(self) -> int:
        return hash(json.dumps(self.to_record(), sort_keys=True, separators=(",", ":")))

    @property
    def source_shape(self) -> tuple[int, int]:
        return (int(self.decoded_shape[0]), int(self.decoded_shape[1]))

    def to_record(self) -> dict:
        transform = self.transform
        return {
            "schema": self.schema, "frame_id": self.frame_id,
            "source_label": self.source_label, "byte_sha256": self.byte_sha256,
            "byte_length": self.byte_length,
            "decoded_shape": [int(value) for value in self.decoded_shape],
            "decoded_dtype": self.decoded_dtype,
            "decoded_pixel_sha256": self.decoded_pixel_sha256,
            "transform": {
                "source_size": [int(value) for value in transform.source_size],
                "crop_origin": [int(value) for value in transform.crop_origin],
                "crop_size": (None if transform.crop_size is None
                              else [int(value) for value in transform.crop_size]),
                "output_size": (None if transform.output_size is None
                                else [int(value) for value in transform.output_size]),
            },
            "decoder": self.decoder,
        }

    @classmethod
    def from_record(cls, payload: object) -> "FrameIngressProvenanceV1":
        """Rebuild a pin from a canonical record.

        Every integer field is validated as a canonical JSON ``int``: a
        ``bool``, ``str`` or fractional ``float`` is refused rather than
        coerced, so a tampered record cannot rebuild a different pin from the
        bytes it claims.
        """
        if type(payload) is not dict:
            raise IngressProvenanceError(INGRESS_PROVENANCE_UNAVAILABLE, "record")
        transform_payload = payload.get("transform")
        if type(transform_payload) is not dict:
            raise IngressProvenanceError(FRAME_TRANSFORM_UNAVAILABLE, "transform")

        def pair(value: object, *, required: bool = False) -> tuple[int, int] | None:
            if value is None:
                if required:
                    raise IngressProvenanceError(
                        FRAME_TRANSFORM_UNAVAILABLE, "transform")
                return None
            if type(value) is not list or len(value) != 2:
                raise IngressProvenanceError(FRAME_TRANSFORM_UNAVAILABLE, "transform")
            return (
                _canonical_int(value[0], FRAME_TRANSFORM_UNAVAILABLE, "transform"),
                _canonical_int(value[1], FRAME_TRANSFORM_UNAVAILABLE, "transform"))

        transform = FrameTransform(
            source_size=pair(transform_payload.get("source_size"), required=True),
            crop_origin=pair(transform_payload.get("crop_origin")) or (0, 0),
            crop_size=pair(transform_payload.get("crop_size")),
            output_size=pair(transform_payload.get("output_size")))
        decoded_shape = payload["decoded_shape"]
        if type(decoded_shape) not in (list, tuple):
            raise IngressProvenanceError(
                INGRESS_PROVENANCE_UNAVAILABLE, "decoded_shape")
        return cls(
            frame_id=payload["frame_id"], source_label=payload["source_label"],
            byte_sha256=payload["byte_sha256"], byte_length=payload["byte_length"],
            decoded_shape=tuple(
                _canonical_int(value, INGRESS_PROVENANCE_UNAVAILABLE, "decoded_shape")
                for value in decoded_shape),
            decoded_dtype=payload["decoded_dtype"],
            decoded_pixel_sha256=payload["decoded_pixel_sha256"],
            transform=transform, schema=payload["schema"], decoder=payload["decoder"])


def build_frame_ingress_provenance(
    *,
    frame_id: str,
    source_label: str,
    encoded_bytes: bytes,
    decoded_pixels: np.ndarray,
    transform: FrameTransform,
    decoded_pixel_sha256: str,
) -> FrameIngressProvenanceV1:
    """Build one ingress pin from the single shared encoded-byte buffer.

    ``encoded_bytes`` must be the exact buffer the decoder consumed; the byte
    digest is computed here from those bytes (never from a second ``_sha256``
    read of ``source_path``).
    """
    if type(encoded_bytes) is not bytes or not encoded_bytes:
        raise IngressProvenanceError(INGRESS_PROVENANCE_UNAVAILABLE, "encoded_bytes")
    pixels = np.asarray(decoded_pixels)
    if pixels.dtype != np.uint8 or pixels.ndim not in (2, 3):
        raise IngressProvenanceError(INGRESS_PROVENANCE_UNAVAILABLE, "decoded_pixels")
    if not _hex64(decoded_pixel_sha256):
        raise IngressProvenanceError(
            INGRESS_PROVENANCE_UNAVAILABLE, "decoded_pixel_sha256")
    return FrameIngressProvenanceV1(
        frame_id=frame_id, source_label=source_label,
        byte_sha256=hashlib.sha256(encoded_bytes).hexdigest(),
        byte_length=len(encoded_bytes),
        decoded_shape=tuple(int(value) for value in pixels.shape),
        decoded_dtype=pixels.dtype.str,
        decoded_pixel_sha256=decoded_pixel_sha256, transform=transform)
