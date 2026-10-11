"""Evaluation-only public projection of retained reading-region evidence."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import numpy as np

from mc_photogrammetry.reconstruction import lattice_planes

SCHEMA = "mc-photogrammetry-reading-region-mask-v1"
MAX_READING_REGION_PIXELS = 6_000_000


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ReadingRegionMask:
    schema: str
    frame_id: str
    canonical_reading_index: int
    source_region_id: str
    height: int
    width: int
    mask_bool_bytes: bytes
    source_evidence_sha256: str
    mask_sha256: str = ""

    def __post_init__(self) -> None:
        if (self.schema != SCHEMA or not self.frame_id
                or type(self.canonical_reading_index) is not int
                or self.canonical_reading_index < 0
                or len(self.source_region_id) != 64
                or type(self.height) is not int or self.height <= 0
                or type(self.width) is not int or self.width <= 0
                or type(self.mask_bool_bytes) is not bytes
                or len(self.mask_bool_bytes) != self.height * self.width
                or any(value not in (0, 1) for value in self.mask_bool_bytes)
                or len(self.source_evidence_sha256) != 64):
            raise ValueError("reading region mask is malformed")
        digest = _sha([SCHEMA, self.frame_id, self.canonical_reading_index,
                       self.source_region_id, self.height, self.width,
                       hashlib.sha256(self.mask_bool_bytes).hexdigest(),
                       self.source_evidence_sha256])
        if self.mask_sha256 not in ("", digest):
            raise ValueError("reading region mask digest mismatch")
        object.__setattr__(self, "mask_sha256", digest)

    @property
    def mask(self) -> np.ndarray:
        return np.frombuffer(self.mask_bool_bytes, dtype=np.bool_).reshape(
            self.height, self.width)

    def to_record(self) -> dict:
        return {"schema": self.schema, "frame_id": self.frame_id,
                "canonical_reading_index": self.canonical_reading_index,
                "source_region_id": self.source_region_id,
                "height": self.height, "width": self.width,
                "mask_bytes_sha256": hashlib.sha256(self.mask_bool_bytes).hexdigest(),
                "source_evidence_sha256": self.source_evidence_sha256,
                "mask_sha256": self.mask_sha256}


def reading_region_mask(
    reading: lattice_planes.PlaneReading,
    *,
    canonical_reading_index: int,
    expected_frame_id: str | None = None,
    expected_source_region_id: str | None = None,
    usable_mask: np.ndarray | None = None,
    max_pixels: int = MAX_READING_REGION_PIXELS,
) -> ReadingRegionMask:
    """Decode one exact retained mask without changing its production record."""
    if not isinstance(reading, lattice_planes.PlaneReading):
        raise ValueError("reading has a foreign type")
    if type(canonical_reading_index) is not int or canonical_reading_index < 0:
        raise ValueError("canonical reading index must be non-negative")
    if expected_frame_id is not None and reading.frame != expected_frame_id:
        raise ValueError("reading frame provenance mismatch")
    evidence = reading.region_evidence
    if evidence is None:
        raise ValueError("reading has no region evidence")
    if (expected_source_region_id is not None
            and evidence.source_region_id != expected_source_region_id):
        raise ValueError("reading source-region provenance mismatch")
    height, width = evidence.image_shape
    if (type(max_pixels) is not int or max_pixels <= 0
            or height * width > max_pixels):
        raise ValueError("reading region mask exceeds pixel cap")
    decoded = lattice_planes._decode_region_evidence(
        evidence, reading.axis, reading.polygon)
    if usable_mask is not None:
        usable = np.asarray(usable_mask)
        if usable.dtype != np.bool_ or usable.shape != decoded.shape:
            raise ValueError("usable mask dimensions or dtype differ")
        decoded = decoded & usable
    source_digest = _sha(evidence.to_record())
    payload = np.ascontiguousarray(decoded, dtype=np.bool_).tobytes(order="C")
    return ReadingRegionMask(
        SCHEMA, reading.frame, canonical_reading_index,
        evidence.source_region_id, height, width, payload, source_digest)
