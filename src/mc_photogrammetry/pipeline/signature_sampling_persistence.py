"""Evaluation-only persistence for the ordinary sampler's own tap witnesses.

Task ``DS-IMG-V2-SAMPLING-WITNESS-PERSISTENCE`` (AGENT_COMMS 2026-10-05). Sol's
read-only review found the exact loss seam: ``derive_image_derived_signatures``
retains the full :class:`~mc_photogrammetry.reconstruction.lattice_planes.SignatureSamplingWitnessV1`
inside every :class:`~mc_photogrammetry.pipeline.candidate_support_provenance.SourceTapWitnessBindingV1`,
but the accepted V1 Route B record writes only ``item.to_record()`` - the binding
identity plus ``witness_sha256`` - so the sampling tap tables are unrecoverable.
This module is the additive, independently versioned, strictly capped JSON
section that persists those exact payloads so a later Phase-B ownership join can
reconstruct them from a saved record **without re-running any screenshot**.

What it is:

* a pure serializer/deserializer plus its own bounded stable-file reader;
* deterministic: arrays are persisted as canonical base64 of their little-endian
  raw bytes with an exact per-field dtype (``<i8``, ``<f4``, ``<u2``, ``|b1``),
  declared shape, byte length and raw SHA-256;
* fail-closed: every cap is checked on plain declared sizes **before** any
  ``tolist``, base64 encode/decode, JSON serialization of arrays or NumPy
  allocation, and every reconstructed witness must reproduce its original
  nested ``witness_sha256``.

What it is not:

* it never reads an image, truth, camera, mask, scene, world, Litematic, JAR or
  evaluator payload and never runs inference, a matcher, a solver or the
  signature sampler;
* it never truncates, silently omits or substitutes an entry: a budget failure
  becomes a small typed ``persistence_refused`` outcome that leaves the original
  bound/refused witness and the matcher disposition untouched, and a missing
  payload stays UNKNOWN;
* it emits no ownership/correctness score and exposes no manual metadata.

The section is embedded by the runner under ``sampling_persistence`` and is
bound into the outer record's ``artifact_sha256`` alongside the unchanged
``CompleteSupportCaptureV2``.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from mc_photogrammetry.pipeline import candidate_support_provenance as provenance
from mc_photogrammetry.reconstruction import lattice
from mc_photogrammetry.reconstruction import lattice_planes

__all__ = [
    "SAMPLING_PERSISTENCE_SCHEMA",
    "SAMPLING_PERSISTENCE_ENTRY_SCHEMA",
    "SAMPLING_PERSISTENCE_DIGEST_DOMAIN",
    "SAMPLING_PERSISTENCE_STATUS_COMPLETE",
    "SAMPLING_PERSISTENCE_STATUS_REFUSED",
    "SAMPLING_PERSISTENCE_OUTCOME_BOUND",
    "SAMPLING_PERSISTENCE_OUTCOME_REFUSED",
    "SAMPLING_PERSISTENCE_OUTCOME_PERSISTENCE_REFUSED",
    "SAMPLING_PERSISTENCE_STATE_AVAILABLE",
    "SAMPLING_PERSISTENCE_STATE_UNAVAILABLE",
    "SAMPLING_PERSISTENCE_SCHEMA_VERSION",
    "SAMPLING_PERSISTENCE_LEGACY_SCHEMA",
    "SAMPLING_ARRAY_DTYPES",
    "SAMPLING_PERSISTENCE_LIMITS",
    "SamplingPersistenceError",
    "SamplingPersistenceLimitsV1",
    "PersistenceAccountingV1",
    "SamplingPersistenceEntryV1",
    "SignatureSamplingPersistenceSectionV1",
    "persist_signature_witnesses",
    "refused_section_record",
    "validate_section_plain",
    "decode_section",
    "cross_bind_witness_summaries",
    "section_digest",
    "canonical_json_size",
    "scan_json_bounds",
    "read_bounded_record_bytes",
]

# --- Schema and vocabulary ----------------------------------------------------

SAMPLING_PERSISTENCE_SCHEMA = "mc-photogrammetry-signature-sampling-persistence-v1"
SAMPLING_PERSISTENCE_ENTRY_SCHEMA = (
    "mc-photogrammetry-signature-sampling-persistence-entry-v1")
SAMPLING_PERSISTENCE_DIGEST_DOMAIN = "signature_sampling_persistence"
# A V1 record never carried this section; the preflight keeps reading it as
# legacy with the sampling payload typed unavailable.
SAMPLING_PERSISTENCE_LEGACY_SCHEMA = "mc-photogrammetry-route-b-diagnostic-record-v1"
SAMPLING_PERSISTENCE_SCHEMA_VERSION = "mc-photogrammetry-route-b-diagnostic-record-v2"

SAMPLING_PERSISTENCE_STATUS_COMPLETE = "complete"
SAMPLING_PERSISTENCE_STATUS_REFUSED = "refused"

SAMPLING_PERSISTENCE_OUTCOME_BOUND = "bound"
SAMPLING_PERSISTENCE_OUTCOME_REFUSED = "refused"
SAMPLING_PERSISTENCE_OUTCOME_PERSISTENCE_REFUSED = "persistence_refused"

SAMPLING_PERSISTENCE_STATE_AVAILABLE = "available"
SAMPLING_PERSISTENCE_STATE_UNAVAILABLE = "unavailable"

# The exact serialized dtype of every persisted array. Only these four strings
# are accepted; NumPy aliases (``int64``, ``f4``, ``=i8``, ``>i8``, ...) are
# deliberately NOT broadened into them and an unvalidated string never reaches
# ``np.dtype``.
SAMPLING_ARRAY_DTYPES = ("<i8", "<f4", "<u2", "|b1")

# field -> (exact dtype, ndim, itemsize). The itemsize is pinned here so the
# plain caps/structural validator never has to call ``np.dtype`` on any string.
_ARRAY_SPECS: dict[str, tuple[str, int, int]] = {
    "cell_source_ptr": ("<i8", 1, 8),
    "cell_source_indices": ("<i8", 1, 8),
    "map_x": ("<f4", 2, 4),
    "map_y": ("<f4", 2, 4),
    "depth_valid": ("|b1", 2, 1),
    "tap_ptr": ("<i8", 1, 8),
    "tap_source_indices": ("<i8", 1, 8),
    "tap_weight_numerators": ("<u2", 1, 2),
}

_RECTIFICATION_ARRAY_FIELDS = (
    "map_x", "map_y", "depth_valid", "tap_ptr", "tap_source_indices",
    "tap_weight_numerators")

# --- Typed refusal reasons ----------------------------------------------------

REASON_PERSISTENCE_ENTRY_CAP = "persistence_entry_cap"
REASON_PERSISTENCE_ENTRY_PAYLOAD_OVER_CAP = "persistence_entry_payload_over_cap"
REASON_PERSISTENCE_TOTAL_PAYLOAD_OVER_CAP = "persistence_total_payload_over_cap"
REASON_PERSISTENCE_ENTRY_METADATA_OVER_CAP = "persistence_entry_metadata_over_cap"
REASON_PERSISTENCE_TOTAL_METADATA_OVER_CAP = "persistence_total_metadata_over_cap"
REASON_PERSISTENCE_SERIALIZED_OVER_CAP = "persistence_serialized_over_cap"
REASON_PERSISTENCE_WORK_OVER_CAP = "persistence_work_over_cap"
REASON_PERSISTENCE_DEPTH_OVER_CAP = "persistence_depth_over_cap"
REASON_PERSISTENCE_WITNESS_UNPERSISTABLE = "persistence_witness_unpersistable"
REASON_PERSISTENCE_SCHEMA_INVALID = "persistence_schema_invalid"
REASON_PERSISTENCE_POLICY_MISMATCH = "persistence_policy_mismatch"
REASON_PERSISTENCE_RECORD_MALFORMED = "persistence_record_malformed"
REASON_PERSISTENCE_ARRAY_OVER_CAP = "persistence_array_over_cap"
REASON_PERSISTENCE_DTYPE_INVALID = "persistence_dtype_invalid"
REASON_PERSISTENCE_INDEX_INVALID = "persistence_index_invalid"
REASON_PERSISTENCE_WEIGHT_INVALID = "persistence_weight_invalid"
REASON_PERSISTENCE_RECTANGLE_INVALID = "persistence_rectangle_invalid"
REASON_PERSISTENCE_HASH_MISMATCH = "persistence_hash_mismatch"
REASON_PERSISTENCE_DIGEST_MISMATCH = "persistence_digest_mismatch"
REASON_PERSISTENCE_DUPLICATE_ENTRY = "persistence_duplicate_entry"
REASON_PERSISTENCE_SUMMARY_MISMATCH = "persistence_summary_mismatch"
REASON_PERSISTENCE_INGRESS_MISMATCH = "persistence_ingress_mismatch"
REASON_PERSISTENCE_OUTCOME_INVALID = "persistence_outcome_invalid"
REASON_PERSISTENCE_INPUT_UNREADABLE = "persistence_input_unreadable"
REASON_PERSISTENCE_INPUT_OVER_CAP = "persistence_input_over_cap"
REASON_PERSISTENCE_INPUT_SIZE_DRIFT = "persistence_input_size_drift"
REASON_PERSISTENCE_TOKEN_OVER_CAP = "persistence_token_over_cap"

PERSISTENCE_REFUSAL_REASONS = (
    REASON_PERSISTENCE_ENTRY_CAP, REASON_PERSISTENCE_ENTRY_PAYLOAD_OVER_CAP,
    REASON_PERSISTENCE_TOTAL_PAYLOAD_OVER_CAP,
    REASON_PERSISTENCE_ENTRY_METADATA_OVER_CAP,
    REASON_PERSISTENCE_TOTAL_METADATA_OVER_CAP,
    REASON_PERSISTENCE_SERIALIZED_OVER_CAP, REASON_PERSISTENCE_WORK_OVER_CAP,
    REASON_PERSISTENCE_DEPTH_OVER_CAP, REASON_PERSISTENCE_WITNESS_UNPERSISTABLE,
    REASON_PERSISTENCE_SCHEMA_INVALID, REASON_PERSISTENCE_POLICY_MISMATCH,
    REASON_PERSISTENCE_RECORD_MALFORMED, REASON_PERSISTENCE_ARRAY_OVER_CAP,
    REASON_PERSISTENCE_DTYPE_INVALID, REASON_PERSISTENCE_INDEX_INVALID,
    REASON_PERSISTENCE_WEIGHT_INVALID, REASON_PERSISTENCE_RECTANGLE_INVALID,
    REASON_PERSISTENCE_HASH_MISMATCH, REASON_PERSISTENCE_DIGEST_MISMATCH,
    REASON_PERSISTENCE_DUPLICATE_ENTRY, REASON_PERSISTENCE_SUMMARY_MISMATCH,
    REASON_PERSISTENCE_INGRESS_MISMATCH, REASON_PERSISTENCE_OUTCOME_INVALID,
    REASON_PERSISTENCE_INPUT_UNREADABLE, REASON_PERSISTENCE_INPUT_OVER_CAP,
    REASON_PERSISTENCE_INPUT_SIZE_DRIFT, REASON_PERSISTENCE_TOKEN_OVER_CAP,
)

# --- Caps ---------------------------------------------------------------------

MIB = 1024 * 1024
READ_CHUNK_BYTES = MIB


class SamplingPersistenceError(ValueError):
    """Typed bounded refusal raised by this module's validators."""

    def __init__(self, reason: str, field: str = "", detail: str = "") -> None:
        if reason not in PERSISTENCE_REFUSAL_REASONS:
            reason = REASON_PERSISTENCE_RECORD_MALFORMED
        self.reason = reason
        self.field = field
        self.detail = detail
        super().__init__(reason if not detail else f"{reason}: {detail}")


@dataclass(frozen=True, slots=True)
class SamplingPersistenceLimitsV1:
    """The single reviewed cap profile, validated as a configuration.

    ``to_record``/``from_record`` bind the exact profile into the section digest,
    so a record can only be decoded with the profile it declares (and the caller
    additionally passes this reviewed profile explicitly, never the record's own).
    """

    max_entries: int = 85
    max_entry_payload_bytes: int = 2 * MIB
    max_total_payload_bytes: int = 8 * MIB
    max_entry_metadata_bytes: int = 64 * 1024
    max_total_metadata_bytes: int = 512 * 1024
    max_serialized_bytes: int = 32 * MIB
    max_depth: int = 16
    max_tokens: int = 6_000_000
    max_work_units: int = 20_000_000
    max_cells_per_side: int = lattice_planes.MAX_SIGNATURE_CELLS
    max_rectified_side: int = lattice_planes.MAX_RECTIFIED_SIDE
    # The *source image* the rectification was sampled from is a different
    # quantity from the rectified patch side and must never be squeezed into
    # ``max_rectified_side``: the known T2 ingress is 2560x1369, so a 2048 side
    # bound refused valid witnesses. Source shapes are additionally bound to the
    # pinned ingress shape on every decode that carries root pins (see
    # ``decode_section``), so these standalone caps are structural only.
    max_source_side: int = 4096
    max_source_pixels: int = 4096 * 4096

    def __post_init__(self) -> None:
        values = (
            self.max_entries, self.max_entry_payload_bytes,
            self.max_total_payload_bytes, self.max_entry_metadata_bytes,
            self.max_total_metadata_bytes, self.max_serialized_bytes,
            self.max_depth, self.max_tokens, self.max_work_units,
            self.max_cells_per_side, self.max_rectified_side,
            self.max_source_side, self.max_source_pixels)
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("sampling persistence caps must be positive integers")
        if self.max_entry_payload_bytes > self.max_total_payload_bytes:
            raise ValueError("per-entry payload cap exceeds the aggregate cap")
        if self.max_entry_metadata_bytes > self.max_total_metadata_bytes:
            raise ValueError("per-entry metadata cap exceeds the aggregate cap")
        if self.max_entry_metadata_bytes > self.max_serialized_bytes:
            raise ValueError("per-entry metadata cap exceeds the serialized cap")
        if self.max_depth < 4:
            raise ValueError("sampling persistence depth cap is too small")
        maximum_side = lattice_planes.MAX_RECTIFIED_SIDE
        if self.max_cells_per_side > maximum_side:
            raise ValueError("cells-per-side cap exceeds the rectified side cap")
        if self.max_rectified_side > maximum_side:
            raise ValueError("rectified-side cap exceeds the producer cap")
        if self.max_source_pixels > self.max_source_side * self.max_source_side:
            raise ValueError("source pixel cap exceeds the source-side cap squared")

    def to_record(self) -> dict:
        return {
            "max_entries": self.max_entries,
            "max_entry_payload_bytes": self.max_entry_payload_bytes,
            "max_total_payload_bytes": self.max_total_payload_bytes,
            "max_entry_metadata_bytes": self.max_entry_metadata_bytes,
            "max_total_metadata_bytes": self.max_total_metadata_bytes,
            "max_serialized_bytes": self.max_serialized_bytes,
            "max_depth": self.max_depth,
            "max_tokens": self.max_tokens,
            "max_work_units": self.max_work_units,
            "max_cells_per_side": self.max_cells_per_side,
            "max_rectified_side": self.max_rectified_side,
            "max_source_side": self.max_source_side,
            "max_source_pixels": self.max_source_pixels,
        }

    @classmethod
    def from_record(cls, record: object) -> "SamplingPersistenceLimitsV1":
        fields = {
            "max_entries", "max_entry_payload_bytes", "max_total_payload_bytes",
            "max_entry_metadata_bytes", "max_total_metadata_bytes",
            "max_serialized_bytes", "max_depth", "max_tokens", "max_work_units",
            "max_cells_per_side", "max_rectified_side", "max_source_side",
            "max_source_pixels"}
        record = _as_dict(record, "policy")
        if set(record) != fields:
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_RECORD_MALFORMED, "policy")
        if any(type(record[name]) is not int for name in fields):
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_RECORD_MALFORMED, "policy")
        try:
            return cls(**record)
        except ValueError as error:
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_RECORD_MALFORMED, "policy",
                type(error).__name__) from error


SAMPLING_PERSISTENCE_LIMITS = SamplingPersistenceLimitsV1()


# --- Canonical helpers --------------------------------------------------------

def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _hex64(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _is_index(value: object) -> bool:
    return type(value) is int and value >= 0


def _is_positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _as_dict(value: object, field: str) -> dict:
    if type(value) is not dict:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECORD_MALFORMED, field)
    return value


def _as_list(value: object, field: str) -> list:
    if type(value) is not list:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECORD_MALFORMED, field)
    return value


def _require_keys(record: dict, required: Iterable[str], optional: Iterable[str],
                  field: str) -> None:
    required = set(required)
    optional = set(optional)
    keys = set(record)
    if not required <= keys or not keys <= (required | optional):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECORD_MALFORMED, field)


def _shape_product(shape: object, field: str, *, ndim: int) -> int:
    if (type(shape) is not list or len(shape) != ndim
            or any(type(value) is not int or value < 0 for value in shape)):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECORD_MALFORMED, field)
    return _product(shape)


def _product(values: Sequence[int]) -> int:
    product = 1
    for value in values:
        product *= int(value)
    return product


def _escaped_text_size(text: str) -> int:
    """Exact ``ensure_ascii=True`` JSON size of one string body (no quotes)."""
    size = 0
    for character in text:
        code = ord(character)
        if character in '"\\' or character in "\b\f\n\r\t":
            size += 2
        elif code < 0x20:
            size += 6
        elif code < 0x80:
            size += 1
        elif code > 0xFFFF:
            size += 12
        else:
            size += 6
    return size


def canonical_json_size(value: object) -> int:
    """Exact byte length of :func:`_canonical_bytes` for plain JSON data.

    The section validator and the runner need the *declared* serialized size of a
    structure **before** allocating the serialized string, so this walks the plain
    scalar structure and returns the exact canonical JSON length without building
    it. Only the JSON data model is accepted (``dict``/``list``/``str``/``int``/
    ``float``/``bool``/``None``); anything else is a typed malformed refusal.
    """
    if value is None:
        return 4
    if value is True:
        return 4
    if value is False:
        return 5
    if type(value) is int:
        return len(str(value))
    if type(value) is float:
        if value != value or value in (float("inf"), float("-inf")):
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_RECORD_MALFORMED, "json_number")
        return len(repr(value))
    if type(value) is str:
        return 2 + _escaped_text_size(value)
    if type(value) is list:
        return 2 + sum(canonical_json_size(item) for item in value) \
            + max(0, len(value) - 1)
    if type(value) is dict:
        total = 2
        for index, (key, item) in enumerate(value.items()):
            if type(key) is not str:
                raise SamplingPersistenceError(
                    REASON_PERSISTENCE_RECORD_MALFORMED, "json_key")
            if index:
                total += 1
            total += 2 + _escaped_text_size(key) + 1 + canonical_json_size(item)
        return total
    raise SamplingPersistenceError(
        REASON_PERSISTENCE_RECORD_MALFORMED, "json_type")


def _rectangles_json_size(rectangles: Sequence[Sequence[int]]) -> int:
    """Exact canonical JSON size of a rectangle sequence, no list materialization."""
    total = 2
    for index, rectangle in enumerate(rectangles):
        if index:
            total += 1
        total += 2
        for position, value in enumerate(rectangle):
            if position:
                total += 1
            total += len(str(int(value)))
    return total


def _declared_text_bytes(value: object, *, field: str, budget: int) -> int:
    """Exact JSON size of one string, refusing over-budget text before walking it.

    ``escaped`` is never shorter than the raw string, so a raw length above the
    whole metadata budget is an over-cap refusal that is reached without scanning
    the text at all; otherwise the walk is bounded by ``budget``.
    """
    if type(value) is not str:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECORD_MALFORMED, field)
    if len(value) > budget:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_ENTRY_METADATA_OVER_CAP, field, str(len(value)))
    return _escaped_text_size(value)


def _document_depth(value: object, *, cap: int, depth: int = 0,
                    field: str = "section") -> int:
    if depth > cap:
        raise SamplingPersistenceError(REASON_PERSISTENCE_DEPTH_OVER_CAP, field)
    deepest = depth
    if type(value) is dict:
        for item in value.values():
            deepest = max(deepest, _document_depth(
                item, cap=cap, depth=depth + 1, field=field))
    elif type(value) is list:
        for item in value:
            deepest = max(deepest, _document_depth(
                item, cap=cap, depth=depth + 1, field=field))
    return deepest


# --- Bounded stable-file read + prescan ---------------------------------------

def scan_json_bounds(data: bytes) -> tuple[int, int]:
    """Return ``(max_nesting_depth, token_count)`` without allocating objects.

    Strings and escapes are skipped so brackets inside a value never count. The
    scan is one linear pass over the raw bytes and is the depth/work gate that
    runs *before* ``json.loads``.
    """
    depth = 0
    max_depth = 0
    tokens = 0
    in_string = False
    escaped = False
    for byte in data:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:          # backslash
                escaped = True
            elif byte == 0x22:          # quote
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte in (0x7B, 0x5B):      # { [
            depth += 1
            tokens += 1
            if depth > max_depth:
                max_depth = depth
        elif byte in (0x7D, 0x5D):      # } ]
            depth -= 1
            if depth < 0:
                raise SamplingPersistenceError(
                    REASON_PERSISTENCE_RECORD_MALFORMED, "json_structure")
        elif byte in (0x2C, 0x3A):      # , :
            tokens += 1
    if in_string or depth != 0:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECORD_MALFORMED, "json_structure")
    return max_depth, tokens


def _linklike_path(path: Path) -> bool:
    """True for a symlink or a Windows reparse point (including a junction)."""
    try:
        info = path.lstat()
    except OSError:
        return True
    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse)


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return ((left.st_dev, left.st_ino, left.st_size, left.st_mtime_ns)
            == (right.st_dev, right.st_ino, right.st_size, right.st_mtime_ns))


def _read_stream_bounded(stream, max_bytes: int) -> bytes:
    """Read at most ``max_bytes + 1`` bytes from ``stream`` in fixed chunks."""
    chunks = bytearray()
    while len(chunks) <= max_bytes:
        chunk = stream.read(min(READ_CHUNK_BYTES, max_bytes + 1 - len(chunks)))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)


def _read_bounded_file_snapshot(path: Path, max_bytes: int) -> bytes:
    """Read one bounded opened-file snapshot capped at ``max_bytes``.

    Mirrors the reviewed ``runtime_render_provenance`` bounded snapshot contract
    (no-follow/reparse rejection, ``lstat``/``fstat`` identity, chunked reads
    capped at ``max_bytes + 1``, post-read identity re-check) under this module's
    typed refusal taxonomy. The file is read through one opened descriptor, so a
    concurrent grow/replace can never make the reader allocate more than
    ``max_bytes + 1`` bytes.

    Residual limitation (documented, not claimed closed): the identity checks are
    ``(st_dev, st_ino, st_size, st_mtime_ns)`` on the path and the descriptor
    before/after the read. They detect a grow, truncation, replacement or ordinary
    concurrent write, but they cannot prove an *atomic* snapshot against a hostile
    writer that overwrites in place with the same size and restores the previous
    mtime before the final ``lstat``. No descriptor-level snapshot primitive is
    available in this process, so the record is validated as one stable
    regular-file read under a best-effort identity gate, not as a lock.
    """
    path = Path(path)
    if _linklike_path(path):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_INPUT_UNREADABLE, "record", "link_or_reparse_point")
    try:
        before_path = path.lstat()
    except OSError as error:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_INPUT_UNREADABLE, "record",
            type(error).__name__) from error
    if not stat.S_ISREG(before_path.st_mode):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_INPUT_UNREADABLE, "record", "not_regular_file")
    if before_path.st_size > max_bytes:
        raise SamplingPersistenceError(REASON_PERSISTENCE_INPUT_OVER_CAP, "record")
    flags = (os.O_RDONLY | getattr(os, "O_BINARY", 0)
             | getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_INPUT_UNREADABLE, "record",
            type(error).__name__) from error
    try:
        try:
            before_fd = os.fstat(descriptor)
            if not _same_file_identity(before_path, before_fd):
                raise SamplingPersistenceError(
                    REASON_PERSISTENCE_INPUT_SIZE_DRIFT, "record")
            chunks = bytearray()
            while len(chunks) <= max_bytes:
                chunk = os.read(
                    descriptor, min(READ_CHUNK_BYTES, max_bytes + 1 - len(chunks)))
                if not chunk:
                    break
                chunks.extend(chunk)
            raw = bytes(chunks)
            after_fd = os.fstat(descriptor)
        except OSError as error:
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_INPUT_UNREADABLE, "record",
                type(error).__name__) from error
    finally:
        os.close(descriptor)
    try:
        after_path = path.lstat()
    except OSError as error:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_INPUT_SIZE_DRIFT, "record",
            type(error).__name__) from error
    if len(raw) > max_bytes:
        raise SamplingPersistenceError(REASON_PERSISTENCE_INPUT_OVER_CAP, "record")
    if (len(raw) != before_path.st_size
            or not _same_file_identity(before_fd, after_fd)
            or not _same_file_identity(before_path, after_path)):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_INPUT_SIZE_DRIFT, "record")
    return raw


@dataclass(frozen=True, slots=True)
class BoundedRecordSnapshotV1:
    data: bytes
    sha256: str
    byte_length: int
    max_json_depth: int
    json_tokens: int


def read_bounded_record_bytes(
    path: "str | os.PathLike[str]",
    limits: SamplingPersistenceLimitsV1 = SAMPLING_PERSISTENCE_LIMITS,
    *,
    reader: Callable[[object], bytes] | None = None,
) -> BoundedRecordSnapshotV1:
    """Read one record under the declared caps, refusing before allocation.

    By default the record is read through one opened, non-link/reparse regular
    file descriptor whose ``lstat``/``fstat`` identity and size must agree before
    and after a chunked read capped at ``max_serialized_bytes + 1``; a file that
    grows or is replaced mid-read is refused instead of allocated. This is a
    stable-read identity gate on this platform - it is *not* an atomic snapshot
    against a same-size in-place writer that also restores the recorded mtime (see
    the residual limitation on :func:`_read_bounded_file_snapshot`). An injected
    ``reader`` keeps the same metadata pre-cap and post-read byte checks and is
    trusted to return the exact bytes of that one path. Depth and token scans run
    on the raw bytes before ``json.loads``.
    """
    path = Path(path)
    cap = limits.max_serialized_bytes
    if reader is not None:
        try:
            size = path.lstat().st_size
        except OSError as error:
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_INPUT_UNREADABLE, "record",
                type(error).__name__) from error
        if size > cap:
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_INPUT_OVER_CAP, "record")
        try:
            data = reader(path)
        except OSError as error:
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_INPUT_UNREADABLE, "record",
                type(error).__name__) from error
        if type(data) is not bytes:
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_INPUT_UNREADABLE, "record")
        if len(data) != size:
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_INPUT_SIZE_DRIFT, "record")
        if len(data) > cap:
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_INPUT_OVER_CAP, "record")
    else:
        data = _read_bounded_file_snapshot(path, cap)
    max_depth, tokens = scan_json_bounds(data)
    if max_depth > limits.max_depth:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_DEPTH_OVER_CAP, "record")
    if tokens > limits.max_tokens:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_TOKEN_OVER_CAP, "record")
    return BoundedRecordSnapshotV1(
        data=data, sha256=hashlib.sha256(data).hexdigest(), byte_length=len(data),
        max_json_depth=max_depth, json_tokens=tokens)


# --- Array contract -----------------------------------------------------------

def _declare_contract_array(array: object, field: str, *,
                            remaining: int) -> tuple[np.ndarray, int, int]:
    """Return ``(array, size, nbytes)`` for one in-memory contract array.

    This is a pure *declaration*: the dtype, rank, element count and byte count
    are read from the original object and nothing is copied, converted or
    scanned. No ``np.ascontiguousarray``/``astype`` copy and no NumPy allocation
    happens here, so an oversized, aliased or foreign-endian array is refused
    before any copy is made; the contiguous copy is deferred to the encoder,
    which only runs after every declared cap has passed.
    """
    spec = _ARRAY_SPECS.get(field)
    if spec is None:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_DTYPE_INVALID, field)
    required_dtype, required_ndim, itemsize = spec
    if not isinstance(array, np.ndarray):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECORD_MALFORMED, field)
    if array.dtype.str != required_dtype:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_DTYPE_INVALID, f"{field}.dtype")
    if array.ndim != required_ndim:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECORD_MALFORMED, f"{field}.ndim")
    size = int(array.size)
    nbytes = size * int(itemsize)
    if nbytes > remaining:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_ARRAY_OVER_CAP, f"{field}.bytes")
    return array, size, nbytes


def _encode_array_record(canonical: np.ndarray) -> dict:
    raw = canonical.tobytes(order="C")
    return {
        "dtype": canonical.dtype.str,
        "shape": list(canonical.shape),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "b64": base64.b64encode(raw).decode("ascii"),
    }


def _declared_array_bytes(record: object, *, field: str, remaining: int,
                          max_size: int) -> tuple[list, int, int]:
    """Validate one plain array record; return ``(shape, size, nbytes)``.

    Nothing here decodes base64 or touches NumPy: the declared shape, exact
    contract dtype, item count, byte length and the canonical base64 length are
    checked against the remaining budget first.
    """
    spec = _ARRAY_SPECS.get(field)
    if spec is None:
        raise SamplingPersistenceError(REASON_PERSISTENCE_DTYPE_INVALID, field)
    required_dtype, required_ndim, itemsize = spec
    record = _as_dict(record, field)
    _require_keys(record, ("dtype", "shape", "bytes", "sha256", "b64"), (),
                  field)
    if record["dtype"] != required_dtype:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_DTYPE_INVALID, f"{field}.dtype")
    shape = record["shape"]
    size = _shape_product(shape, f"{field}.shape", ndim=required_ndim)
    if size > max_size:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_ARRAY_OVER_CAP, f"{field}.size")
    nbytes = size * itemsize
    if type(record["bytes"]) is not int or record["bytes"] != nbytes:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_ARRAY_OVER_CAP, f"{field}.bytes")
    if nbytes > remaining:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_ARRAY_OVER_CAP, f"{field}.bytes")
    if not _hex64(record["sha256"]):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_HASH_MISMATCH, f"{field}.sha256")
    encoded = record["b64"]
    if type(encoded) is not str:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECORD_MALFORMED, f"{field}.b64")
    if len(encoded) != 4 * ((nbytes + 2) // 3):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_ARRAY_OVER_CAP, f"{field}.b64")
    return shape, size, nbytes


def _decode_array_record(record: object, *, field: str, remaining: int,
                         max_size: int) -> np.ndarray:
    """Decode one bounded array record into an immutable contract array."""
    shape, size, nbytes = _declared_array_bytes(
        record, field=field, remaining=remaining, max_size=max_size)
    record = _as_dict(record, field)
    required_dtype = _ARRAY_SPECS[field][0]
    try:
        raw = base64.b64decode(record["b64"], validate=True)
    except (binascii.Error, ValueError) as error:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_ARRAY_OVER_CAP, f"{field}.b64",
            type(error).__name__) from error
    if (len(raw) != nbytes
            or base64.b64encode(raw).decode("ascii") != record["b64"]):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_ARRAY_OVER_CAP, f"{field}.b64")
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_HASH_MISMATCH, f"{field}.sha256")
    array = np.frombuffer(raw, dtype=np.dtype(required_dtype)).reshape(tuple(shape))
    array.flags.writeable = False
    return array


# --- Rectification accounting / encoding / decoding ---------------------------

def _array_metadata_record(array: np.ndarray, *, digest: bool) -> dict:
    """One array's metadata record; no contiguity copy unless a digest is asked."""
    record = {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "bytes": int(array.size) * int(array.dtype.itemsize),
    }
    record["sha256"] = (
        hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()
        if digest else "0" * 64)
    return record


def _rectification_metadata(rect: lattice.RectificationWitnessV1, *,
                            digest: bool) -> dict:
    record: dict = {
        "schema": rect.schema,
        "opencv_version": rect.opencv_version,
        "opencv_build_sha256": rect.opencv_build_sha256,
        "interpolation": rect.interpolation,
        "border": rect.border,
        "inter_bits": rect.inter_bits,
        "weight_denominator": rect.weight_denominator,
        "source_shape": list(rect.source_shape),
        "witness_sha256": rect.witness_sha256,
    }
    for name in _RECTIFICATION_ARRAY_FIELDS:
        record[name] = _array_metadata_record(getattr(rect, name), digest=digest)
    return record


@dataclass(frozen=True, slots=True)
class _RectificationPreflightV1:
    """One rectification witness whose declared arrays fit the payload budget."""

    field: str
    rectification: lattice.RectificationWitnessV1
    arrays: dict
    nbytes: int
    samples: int
    taps: int
    source_pixels: int


def _preflight_rectification(rect: object, *, field: str,
                             limits: SamplingPersistenceLimitsV1,
                             remaining: int) -> _RectificationPreflightV1:
    """Declare one rectification witness under the caps, before any traversal.

    Source-image shape/pixels are checked against the *source* caps and the
    rectified map/rectangle grid against ``max_rectified_side``; every declared
    array byte is charged from the original arrays' shape/dtype/itemsize. This is
    a declaration only: no array is copied, converted, scanned or hashed here, so
    the whole per-entry payload budget is applied before the encoder ever makes
    the contiguous copy.
    """
    if not isinstance(rect, lattice.RectificationWitnessV1):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECORD_MALFORMED, field)
    source_shape = rect.source_shape
    if (type(source_shape) is not tuple or len(source_shape) != 2
            or any(not _is_positive_int(value) for value in source_shape)
            or any(value > limits.max_source_side for value in source_shape)
            or source_shape[0] * source_shape[1] > limits.max_source_pixels):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECTANGLE_INVALID, f"{field}.source_shape")
    map_shape = rect.map_x.shape
    if (rect.map_x.shape != rect.map_y.shape
            or rect.map_x.shape != rect.depth_valid.shape
            or len(map_shape) != 2
            or any(value <= 0 or value > limits.max_rectified_side
                   for value in map_shape)):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECTANGLE_INVALID, f"{field}.map_shape")
    arrays = {}
    total = 0
    for name in _RECTIFICATION_ARRAY_FIELDS:
        declared, size, nbytes = _declare_contract_array(
            getattr(rect, name), name,
            remaining=limits.max_entry_payload_bytes - total)
        arrays[name] = (declared, size, nbytes)
        total += nbytes
    if total > remaining:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_ENTRY_PAYLOAD_OVER_CAP, field)
    return _RectificationPreflightV1(
        field=field, rectification=rect, arrays=arrays, nbytes=total,
        samples=int(rect.map_x.size), taps=int(arrays["tap_source_indices"][1]),
        source_pixels=int(source_shape[0]) * int(source_shape[1]))


def _validate_bilinear_tap_semantics(
        *, samples: int, depth_valid: np.ndarray, tap_ptr: np.ndarray,
        indices: np.ndarray, weights: np.ndarray,
        field: str, denominator: int) -> None:
    """Validate the exact per-sample bilinear producer contract.

    ``lattice.rectification_witness_v1`` appends the non-zero numerators of
    ``cv2.convertMaps(..., CV_16SC2)`` in ``(0,0) (1,0) (0,1) (1,1)`` order, so
    every output sample owns a contiguous ``tap_ptr`` span whose weights sum to
    exactly ``2 ** 2 * 2 ** inter_bits`` (1024). An invalid/depth-excluded sample
    is written as ``xs = ys = -1``: its single non-zero numerator is 1024 and its
    only tap index is the ``-1`` border sentinel. Each stored weight is in range
    and the pointer table is monotonic (already checked), yet an individually
    valid table can still have a wrong per-sample total; that is refused here.
    """
    if int(tap_ptr.size) != samples + 1:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_INDEX_INVALID, f"{field}.tap_ptr")
    if samples <= 0:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_INDEX_INVALID, f"{field}.tap_ptr")
    starts = tap_ptr[:-1]
    counts = tap_ptr[1:] - starts
    if int(counts.min()) < 1 or int(counts.max()) > 4:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_INDEX_INVALID, f"{field}.tap_ptr.span")
    if int(indices.size) != int(weights.size) \
            or int(indices.size) != int(tap_ptr[-1]):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_INDEX_INVALID, f"{field}.tap_ptr")
    sums = np.add.reduceat(weights, starts)
    if bool(np.any(sums != np.asarray(denominator, dtype=sums.dtype))):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_WEIGHT_INVALID, f"{field}.tap_weight_sums")
    invalid = np.logical_not(np.asarray(depth_valid).reshape(-1))
    if bool(invalid.any()):
        if bool(np.any(counts[invalid] != 1)):
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_INDEX_INVALID, f"{field}.tap_ptr.invalid")
        invalid_starts = starts[invalid]
        if bool(np.any(weights[invalid_starts] != denominator)):
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_WEIGHT_INVALID,
                f"{field}.tap_weight_invalid_sample")
        if bool(np.any(indices[invalid_starts] != -1)):
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_INDEX_INVALID,
                f"{field}.tap_source_indices.invalid")


def _validate_rectification_values(preflight: _RectificationPreflightV1, *,
                                   limits: SamplingPersistenceLimitsV1) -> None:
    """The value-level pointer/index/weight/bilinear checks, after all caps."""
    rect = preflight.rectification
    field = preflight.field
    tap_ptr = preflight.arrays["tap_ptr"][0]
    indices = preflight.arrays["tap_source_indices"][0]
    weights = preflight.arrays["tap_weight_numerators"][0]
    if int(tap_ptr.size) != preflight.samples + 1:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_INDEX_INVALID, f"{field}.tap_ptr")
    _validate_pointer_array(
        tap_ptr, field=f"{field}.tap_ptr", final_length=preflight.taps)
    _validate_index_array(
        indices, field=f"{field}.tap_source_indices",
        source_cells=preflight.source_pixels)
    _validate_weight_array(
        weights, field=f"{field}.tap_weight_numerators",
        denominator=rect.weight_denominator)
    _validate_bilinear_tap_semantics(
        samples=preflight.samples,
        depth_valid=preflight.arrays["depth_valid"][0], tap_ptr=tap_ptr,
        indices=indices, weights=weights, field=field,
        denominator=rect.weight_denominator)




def _validate_pointer_array(array: np.ndarray, *, field: str,
                            final_length: int) -> None:
    if array.ndim != 1 or int(array.size) == 0:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_INDEX_INVALID, field)
    if int(array[0]) != 0 or int(array[-1]) != int(final_length):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_INDEX_INVALID, field)
    if int(array.min()) < 0:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_INDEX_INVALID, field)
    if array.size > 1 and bool(np.any(np.diff(array) < 0)):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_INDEX_INVALID, field)


def _validate_index_array(array: np.ndarray, *, field: str,
                          source_cells: int) -> None:
    if array.ndim != 1:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_INDEX_INVALID, field)
    if array.size:
        if int(array.min()) < -1 or int(array.max()) >= int(source_cells):
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_INDEX_INVALID, field)


def _validate_weight_array(array: np.ndarray, *, field: str,
                           denominator: int) -> None:
    if array.ndim != 1:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_WEIGHT_INVALID, field)
    if array.size:
        if int(array.min()) < 1 or int(array.max()) > int(denominator):
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_WEIGHT_INVALID, field)


def _validate_rectangles(rectangles: object, *, field: str,
                         limits: SamplingPersistenceLimitsV1) -> None:
    if type(rectangles) is not tuple and type(rectangles) is not list:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECTANGLE_INVALID, field)
    if len(rectangles) > limits.max_cells_per_side * limits.max_cells_per_side:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECTANGLE_INVALID, f"{field}.count")
    for index, rectangle in enumerate(rectangles):
        if (type(rectangle) is not tuple and type(rectangle) is not list) \
                or len(rectangle) != 4:
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_RECTANGLE_INVALID, f"{field}.{index}")
        if any(type(value) is not int for value in rectangle):
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_RECTANGLE_INVALID, f"{field}.{index}")
        r0, r1, c0, c1 = rectangle
        if (r0 < 0 or c0 < 0 or r1 < r0 or c1 < c0
                or r1 > limits.max_rectified_side
                or c1 > limits.max_rectified_side):
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_RECTANGLE_INVALID, f"{field}.{index}")


def _encode_rectification(rect: lattice.RectificationWitnessV1) -> dict:
    record: dict = {
        "schema": rect.schema,
        "opencv_version": rect.opencv_version,
        "opencv_build_sha256": rect.opencv_build_sha256,
        "interpolation": rect.interpolation,
        "border": rect.border,
        "inter_bits": rect.inter_bits,
        "weight_denominator": rect.weight_denominator,
        "source_shape": list(rect.source_shape),
        "witness_sha256": rect.witness_sha256,
    }
    for name in _RECTIFICATION_ARRAY_FIELDS:
        canonical = np.ascontiguousarray(getattr(rect, name))
        record[name] = _encode_array_record(canonical)
    return record


def _witness_metadata(witness: lattice_planes.SignatureSamplingWitnessV1, *,
                      digest: bool,
                      materialize_rectangles: bool = True) -> dict:
    """The witness record with every array's ``b64`` payload removed.

    ``materialize_rectangles=False`` substitutes empty rectangle sequences, so the
    declared metadata size can be measured from the rectangle cardinalities (see
    :func:`_declared_witness_metadata_bytes`) without ever building the per-cell
    lists. ``digest=False`` never touches an array buffer.
    """
    rectifications = {
        "rgb_rectification": _rectification_metadata(
            witness.rgb_rectification, digest=digest),
        "coverage_rectification": _rectification_metadata(
            witness.coverage_rectification, digest=digest),
        "region_rectification": (None if witness.region_rectification is None
                                 else _rectification_metadata(
                                     witness.region_rectification, digest=digest)),
    }
    record = {
        "schema": witness.schema,
        "grayscale_sha256": witness.grayscale_sha256,
        "gradient_y_sha256": witness.gradient_y_sha256,
        "gradient_x_sha256": witness.gradient_x_sha256,
        "energy_sha256": witness.energy_sha256,
        "coverage_patch_sha256": witness.coverage_patch_sha256,
        "gradient_edge_order": witness.gradient_edge_order,
        "subsample_factor": witness.subsample_factor,
        "seam_band": witness.seam_band,
        "cell_rectangles": ([list(value) for value in witness.cell_rectangles]
                            if materialize_rectangles else []),
        "interior_rectangles": ([list(value) for value in witness.interior_rectangles]
                               if materialize_rectangles else []),
        "witness_sha256": witness.witness_sha256,
    }
    record.update(rectifications)
    for name in ("cell_source_ptr", "cell_source_indices"):
        record[name] = _array_metadata_record(getattr(witness, name), digest=digest)
    return record


def _declared_witness_metadata_bytes(
        witness: lattice_planes.SignatureSamplingWitnessV1) -> int:
    """Exact metadata size from declared cardinalities, before any list build."""
    record = _witness_metadata(witness, digest=False, materialize_rectangles=False)
    total = canonical_json_size(record)
    total += _rectangles_json_size(witness.cell_rectangles) - 2
    total += _rectangles_json_size(witness.interior_rectangles) - 2
    return total


@dataclass(frozen=True, slots=True)
class WitnessAccountingV1:
    payload_bytes: int
    payload_elements: int
    metadata_bytes: int
    work_units: int


def _declared_witness_work(payload_bytes: int, metadata_bytes: int,
                           rectangles: int, interiors: int,
                           array_count: int) -> int:
    """Precharged work for one witness, in declared traversal/encoding units.

    Work is charged on declared size before any walk: one unit per declared
    payload byte (every scan, reduction, hash, base64 encode/decode and NumPy
    rebuild touches its bytes), one per metadata byte (every field is serialized),
    four per rectangle (two rectangle sequences are walked and JSON-sized) and two
    per declared array record. ``validate_section_plain`` recomputes the identical
    charge from the plain declared sizes, so the persisted ``work_units`` and the
    decoded accounting can never disagree. A non-bound outcome has no witness, so
    it is charged separately as one unit plus its exact metadata bytes.
    """
    return (payload_bytes + metadata_bytes + 4 * (rectangles + interiors)
            + 2 * array_count)


def _account_witness(witness: object, *,
                     limits: SamplingPersistenceLimitsV1) -> WitnessAccountingV1:
    """Declare one ordinary witness; never copies, scans, encodes or hashes.

    Order: schema/scalar checks, rectangle bounds, then a *declared* preflight of
    every array (dtype/rank/size/bytes read from the original objects, no copy)
    and of the metadata size from cardinalities (no per-cell list is built); the
    per-entry payload, metadata and work caps are then applied and only then does
    the caller run the pointer/index/weight/bilinear value scans and the encoder's
    contiguous copies (see :func:`_validate_witness_values`). Splitting the
    declaration from the traversal is what keeps a work-budget refusal from
    happening after ``np.ascontiguousarray`` has already copied every array.
    """
    if not isinstance(witness, lattice_planes.SignatureSamplingWitnessV1):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECORD_MALFORMED, "witness")
    if witness.schema != "mc-photogrammetry-signature-sampling-witness-v1":
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_SCHEMA_INVALID, "witness.schema")
    if type(witness.gradient_edge_order) is not int \
            or witness.gradient_edge_order != 1:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_SCHEMA_INVALID, "witness.gradient_edge_order")
    if not _is_positive_int(witness.subsample_factor) \
            or not _is_positive_int(witness.seam_band):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_SCHEMA_INVALID, "witness.sampling_settings")
    for name in ("grayscale_sha256", "gradient_y_sha256", "gradient_x_sha256",
                 "energy_sha256", "coverage_patch_sha256", "witness_sha256"):
        if not _hex64(getattr(witness, name)):
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_HASH_MISMATCH, f"witness.{name}")

    rectangles = witness.cell_rectangles
    interiors = witness.interior_rectangles
    # Both rectangle sequences are type-checked before either is measured with
    # ``len``, so ``interior_rectangles=None`` is a typed refusal, not a raw
    # ``TypeError`` from ``len(None)``.
    _validate_rectangles(rectangles, field="witness.cell_rectangles",
                         limits=limits)
    _validate_rectangles(interiors, field="witness.interior_rectangles",
                         limits=limits)
    if len(interiors) != len(rectangles):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECTANGLE_INVALID, "witness.interior_rectangles")
    if len(rectangles) + 1 != len(witness.cell_source_ptr):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_INDEX_INVALID, "witness.cell_source_ptr")

    # --- Declared preflight: sizes and caps, no traversal and no copy. --------
    total = 0
    elements = 0
    cells: dict = {}
    for name in ("cell_source_ptr", "cell_source_indices"):
        declared, size, nbytes = _declare_contract_array(
            getattr(witness, name), name,
            remaining=limits.max_entry_payload_bytes - total)
        cells[name] = (declared, size, nbytes)
        total += nbytes
        elements += size
    preflights: list[_RectificationPreflightV1] = []
    array_count = 2
    for field, rectification in (
            ("rgb_rectification", witness.rgb_rectification),
            ("coverage_rectification", witness.coverage_rectification),
            ("region_rectification", witness.region_rectification)):
        if rectification is None:
            if field != "region_rectification":
                raise SamplingPersistenceError(
                    REASON_PERSISTENCE_RECORD_MALFORMED, f"witness.{field}")
            continue
        preflight = _preflight_rectification(
            rectification, field=f"witness.{field}", limits=limits,
            remaining=limits.max_entry_payload_bytes - total)
        preflights.append(preflight)
        total += preflight.nbytes
        elements += sum(size for _declared, size, _nbytes
                        in preflight.arrays.values())
        array_count += len(_RECTIFICATION_ARRAY_FIELDS)
    if total > limits.max_entry_payload_bytes:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_ENTRY_PAYLOAD_OVER_CAP, "witness.payload")
    declared_metadata = _declared_witness_metadata_bytes(witness)
    if declared_metadata > limits.max_entry_metadata_bytes:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_ENTRY_METADATA_OVER_CAP, "witness.metadata",
            str(declared_metadata))
    work = _declared_witness_work(
        total, declared_metadata, len(rectangles), len(interiors), array_count)
    if work > limits.max_work_units:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_WORK_OVER_CAP, "witness.work", str(work))
    return WitnessAccountingV1(
        payload_bytes=total, payload_elements=elements,
        metadata_bytes=declared_metadata, work_units=work)


def _validate_witness_values(witness: lattice_planes.SignatureSamplingWitnessV1,
                             *,
                             limits: SamplingPersistenceLimitsV1) -> None:
    """The pointer/index/weight/bilinear value scans for one declared witness.

    Called only after :func:`_account_witness` and the caller's aggregate payload,
    metadata, work and serialized caps have passed, so no scan can run (and no
    NumPy temporary can be allocated) for a witness that is already over budget.
    """
    cells: dict = {}
    total = 0
    for name in ("cell_source_ptr", "cell_source_indices"):
        declared, size, nbytes = _declare_contract_array(
            getattr(witness, name), name,
            remaining=limits.max_entry_payload_bytes - total)
        cells[name] = (declared, size, nbytes)
        total += nbytes
    source_cells = 1
    for field, rectification in (
            ("rgb_rectification", witness.rgb_rectification),
            ("coverage_rectification", witness.coverage_rectification),
            ("region_rectification", witness.region_rectification)):
        if rectification is None:
            continue
        preflight = _preflight_rectification(
            rectification, field=f"witness.{field}", limits=limits,
            remaining=limits.max_entry_payload_bytes - total)
        total += preflight.nbytes
        source_cells = max(source_cells, preflight.source_pixels)
        _validate_rectification_values(preflight, limits=limits)
    pointer, _pointer_size, _pointer_bytes = cells["cell_source_ptr"]
    indices, index_size, _index_bytes = cells["cell_source_indices"]
    _validate_pointer_array(
        pointer, field="witness.cell_source_ptr", final_length=index_size)
    _validate_index_array(
        indices, field="witness.cell_source_indices", source_cells=source_cells)


def _encode_witness(witness: lattice_planes.SignatureSamplingWitnessV1) -> dict:
    record = _witness_metadata(witness, digest=True)
    record["rgb_rectification"] = _encode_rectification(witness.rgb_rectification)
    record["coverage_rectification"] = _encode_rectification(
        witness.coverage_rectification)
    record["region_rectification"] = (
        None if witness.region_rectification is None
        else _encode_rectification(witness.region_rectification))
    for name in ("cell_source_ptr", "cell_source_indices"):
        record[name] = _encode_array_record(
            np.ascontiguousarray(getattr(witness, name)))
    return record



# --- Identity and ordered entries ---------------------------------------------

_IDENTITY_FIELDS = (
    "frame_id", "canonical_index", "alternative_index", "signature_sha256",
    "ingress_sha256", "mask_sha256", "masks_sha256", "max_cells",
    "max_rectified_side", "pixels_per_cell", "witness_sha256")


def _binding_identity(binding: provenance.SourceTapWitnessBindingV1) -> dict:
    return {
        "frame_id": binding.frame_id,
        "canonical_index": binding.canonical_index,
        "alternative_index": binding.alternative_index,
        "signature_sha256": binding.signature_sha256,
        "ingress_sha256": _digest(binding.ingress.to_record()),
        "mask_sha256": binding.mask_sha256,
        "masks_sha256": binding.masks_sha256,
        "max_cells": binding.max_cells,
        "max_rectified_side": binding.max_rectified_side,
        "pixels_per_cell": binding.pixels_per_cell,
        "witness_sha256": binding.witness.witness_sha256,
    }


def _refusal_identity(refusal: provenance.SourceTapWitnessRefusalV1) -> dict:
    return {
        "frame_id": refusal.frame_id,
        "canonical_index": refusal.canonical_index,
        "alternative_index": refusal.alternative_index,
    }


def _persistence_refused_entry(ordinal: int, identity: dict, *,
                               reason: str, detail: str = "") -> dict:
    entry = {
        "outcome": SAMPLING_PERSISTENCE_OUTCOME_PERSISTENCE_REFUSED,
        "ordinal": ordinal,
        "reason": reason,
        "detail": detail,
    }
    entry.update(identity)
    return entry


def _section_payload(section: dict) -> dict:
    return {key: value for key, value in section.items() if key != "section_sha256"}


def section_digest(section: dict) -> str:
    """The canonical digest of one section, excluding its own digest field."""
    return _digest({"domain": SAMPLING_PERSISTENCE_DIGEST_DOMAIN,
                    "payload": _section_payload(section)})


def _seal_section(section: dict) -> dict:
    section = dict(section)
    section["section_sha256"] = section_digest(section)
    return section


# A canonical SHA-256 hex digest is always exactly 64 lowercase hex characters, so
# a placeholder of the same length has the same canonical JSON length as the real
# digest and lets the sealed size be probed before the digest exists.
_SECTION_DIGEST_PLACEHOLDER = "0" * 64


def _sealed_section_size(section: dict) -> int:
    """Exact canonical size of ``section`` once :func:`_seal_section` seals it.

    The seal appends exactly one member, ``"section_sha256":"<64 hex>"``; the
    placeholder digest has the same canonical length as the real one, so this
    walks the unsealed payload and adds that member's exact canonical
    contribution without computing the digest, building a second mapping or
    serializing any JSON: one comma, the quoted key, a colon and the quoted
    64-character digest (1 + 16 + 1 + 66 = 84 bytes). An ad-hoc 80-byte
    reservation under-counted that member by four bytes, which let a 1,085-byte
    complete section pass the declared gate at ``max_serialized_bytes=1084`` and
    reach :func:`_canonical_bytes` before degrading.
    """
    return (canonical_json_size(section)
            + 1
            + canonical_json_size("section_sha256")
            + 1
            + canonical_json_size(_SECTION_DIGEST_PLACEHOLDER))


def _refused_section(limits: SamplingPersistenceLimitsV1, reason: str,
                     field: str = "", detail: str = "") -> dict:
    return _seal_section({
        "schema": SAMPLING_PERSISTENCE_SCHEMA,
        "status": SAMPLING_PERSISTENCE_STATUS_REFUSED,
        "reason": reason,
        "field": field,
        "detail": detail,
        "policy": limits.to_record(),
        "entry_count": 0,
        "entries": [],
        "payload_bytes": 0,
        "metadata_bytes": 0,
        "work_units": 0,
    })


def refused_section_record(
    reason: str, *,
    limits: SamplingPersistenceLimitsV1 = SAMPLING_PERSISTENCE_LIMITS,
    field: str = "", detail: str = "",
) -> dict:
    """A small typed, digest-sealed section that persists no payload.

    The runner uses this when the complete serialized record would exceed its own
    byte cap: the additive sampling section degrades to an explicit typed
    refusal while the support capture and every disposition stay unchanged.
    """
    if reason not in PERSISTENCE_REFUSAL_REASONS:
        reason = REASON_PERSISTENCE_WITNESS_UNPERSISTABLE
    return _refused_section(limits, reason, field, detail)


def _entry_serialized_estimate(metadata_bytes: int, payload_bytes: int) -> int:
    # 4/3 base64 expansion plus a fixed per-entry JSON overhead.
    return metadata_bytes + 4 * ((payload_bytes + 2) // 3) + 2048


def _entry_metadata_bytes(entry: dict, *, witness_metadata: dict | None) -> int:
    """Exact metadata bytes of one entry, with the witness ``b64`` payloads removed."""
    projection = {key: value for key, value in entry.items() if key != "witness"}
    if witness_metadata is not None:
        projection["witness"] = witness_metadata
    return canonical_json_size(projection)


def _declared_entry_metadata_bytes(
        entry: dict, witness: lattice_planes.SignatureSamplingWitnessV1) -> int:
    """Exact bound-entry metadata size from cardinalities, no rectangle lists.

    The witness metadata is measured with its rectangle sequences empty and the
    exact ``_rectangles_json_size`` of each declared sequence is added back
    (replacing an empty list of canonical size 2 changes the total by exactly
    ``_rectangles_json_size - 2``). The per-entry metadata budget is therefore
    applied without ever materializing a per-cell rectangle list; only the
    encoder builds those lists, and only after every applicable budget passed.
    """
    witness_metadata = _witness_metadata(witness, digest=False,
                                         materialize_rectangles=False)
    total = _entry_metadata_bytes(entry, witness_metadata=witness_metadata)
    total += _rectangles_json_size(witness.cell_rectangles) - 2
    total += _rectangles_json_size(witness.interior_rectangles) - 2
    return total


def _non_bound_entry_metadata_bytes(entry: dict, *, limit: int) -> int:
    """Declared metadata bytes of a refused/persistence-refused entry.

    Every bounded string (identity, refusal reason and detail) is length-gated
    before it is walked, so an over-cap detail is refused without serializing it.
    """
    for name in ("frame_id", "reason", "detail"):
        if name in entry:
            _declared_text_bytes(entry[name], field=f"entry.{name}", budget=limit)
    return canonical_json_size(entry)


def persist_signature_witnesses(
    items: Sequence[object],
    *,
    limits: SamplingPersistenceLimitsV1 = SAMPLING_PERSISTENCE_LIMITS,
) -> dict:
    """Return the capped, digest-sealed sampling-persistence section record.

    One ordered outcome is emitted per input item, preserving the item's ordinal
    and identity. Every budget is charged from declared sizes before any copy,
    traversal or serialization. A per-entry budget failure becomes a typed
    ``persistence_refused`` entry - the original bound/refused witness and the
    matcher disposition are never changed - and a malformed witness becomes a
    typed refusal rather than an exception or a truncated payload. Refusal
    reasons, details and identities are part of the metadata budget; an over-cap
    refusal detail degrades to a small typed refused record instead of being
    serialized.

    Work accounting has two distinct counters. ``total_work`` is the serialized
    accounting written to the section and recomputed identically by
    ``validate_section_plain``: a bound entry charges its declared witness work
    and a non-bound outcome charges one declared unit plus its exact metadata
    bytes (the metadata is traversed and JSON-encoded). ``reserved_work`` is
    budget-only: it holds the declared work of a witness whose value scans were
    attempted but refused semantically, so that work is never refunded and a later
    witness cannot spend the same budget. Every work check uses
    ``total_work + reserved_work``, which is always greater than or equal to the
    serialized total, so the producer can never emit a complete section its own
    validator would reject.
    """
    items = tuple(items)
    if len(items) > limits.max_entries:
        return _refused_section(limits, REASON_PERSISTENCE_ENTRY_CAP,
                                "entry_count", str(len(items)))
    entries: list[dict] = []
    total_payload = 0
    total_metadata = 0
    total_work = 0
    reserved_work = 0
    estimated = 512

    def charge_refusal(entry: dict, metadata: int, *,
                       reserved: bool = False) -> "dict | None":
        """Charge one non-payload outcome against every remaining budget.

        Refused and persistence-refused outcomes cost one declared work unit
        *plus* their exact metadata bytes, so a large refusal detail can never
        validate complete under a work cap it exceeds. The entry is checked
        against the aggregate metadata, per-entry metadata *and* aggregate work
        caps before it is appended, so a refusal branch can never bypass a
        remaining budget or emit a complete section that its own
        ``validate_section_plain`` would reject. When ``reserved`` is true the
        witness's declared work was already charged to the budget before its value
        scans; the serialized charge is carved out of that reservation and the
        remainder stays charged (a semantic failure never refunds declared work).
        Returns ``None`` when the entry was appended, otherwise the typed
        whole-section refusal to return.
        """
        nonlocal total_metadata, total_work, reserved_work, estimated
        if total_metadata + metadata > limits.max_total_metadata_bytes:
            return _refused_section(
                limits, REASON_PERSISTENCE_TOTAL_METADATA_OVER_CAP,
                "metadata_bytes")
        if metadata > limits.max_entry_metadata_bytes:
            return _refused_section(
                limits, REASON_PERSISTENCE_ENTRY_METADATA_OVER_CAP, "entry")
        charge = 1 + metadata
        if reserved:
            if charge > reserved_work:
                # Defensive: the serialized charge is normally far below the
                # witness's declared work. If it ever were not, charge only the
                # excess against the remaining budget.
                excess = charge - reserved_work
                if total_work + reserved_work + excess > limits.max_work_units:
                    return _refused_section(
                        limits, REASON_PERSISTENCE_WORK_OVER_CAP, "work_units")
                reserved_work = 0
            else:
                reserved_work -= charge
        elif total_work + reserved_work + charge > limits.max_work_units:
            return _refused_section(
                limits, REASON_PERSISTENCE_WORK_OVER_CAP, "work_units")
        total_metadata += metadata
        total_work += charge
        estimated += _entry_serialized_estimate(metadata, 0)
        entries.append(entry)
        return None

    for ordinal, item in enumerate(items):
        if isinstance(item, provenance.SourceTapWitnessRefusalV1):
            identity = _refusal_identity(item)
            entry = {
                "outcome": SAMPLING_PERSISTENCE_OUTCOME_REFUSED,
                "ordinal": ordinal,
                "reason": item.reason,
                "detail": item.detail,
                **identity,
            }
            degrade = ""
            try:
                metadata = _non_bound_entry_metadata_bytes(
                    entry, limit=limits.max_entry_metadata_bytes)
                if metadata > limits.max_entry_metadata_bytes:
                    degrade = REASON_PERSISTENCE_ENTRY_METADATA_OVER_CAP
                elif total_metadata + metadata > limits.max_total_metadata_bytes:
                    degrade = REASON_PERSISTENCE_TOTAL_METADATA_OVER_CAP
            except SamplingPersistenceError:
                degrade = REASON_PERSISTENCE_ENTRY_METADATA_OVER_CAP
            if degrade:
                entry = _persistence_refused_entry(
                    ordinal, identity, reason=degrade, detail="")
                metadata = _non_bound_entry_metadata_bytes(
                    entry, limit=limits.max_entry_metadata_bytes)
            refused = charge_refusal(entry, metadata)
            if refused is not None:
                return refused
            continue
        if not isinstance(item, provenance.SourceTapWitnessBindingV1):
            return _refused_section(
                limits, REASON_PERSISTENCE_WITNESS_UNPERSISTABLE,
                f"entries.{ordinal}", type(item).__name__)
        identity = _binding_identity(item)

        def refuse(reason: str, detail: str = "") -> tuple[dict, int]:
            entry = _persistence_refused_entry(
                ordinal, identity, reason=reason, detail=detail)
            return entry, _non_bound_entry_metadata_bytes(
                entry, limit=limits.max_entry_metadata_bytes)

        # --- Declared gate: no array copy, scan, list build, encode or hash. ---
        try:
            account = _account_witness(item.witness, limits=limits)
        except SamplingPersistenceError as error:
            if error.reason in (REASON_PERSISTENCE_ENTRY_PAYLOAD_OVER_CAP,
                                REASON_PERSISTENCE_ARRAY_OVER_CAP):
                entry, metadata = refuse(REASON_PERSISTENCE_ENTRY_PAYLOAD_OVER_CAP,
                                         error.reason)
            elif error.reason == REASON_PERSISTENCE_ENTRY_METADATA_OVER_CAP:
                entry, metadata = refuse(REASON_PERSISTENCE_ENTRY_METADATA_OVER_CAP,
                                         error.detail or error.field)
            elif error.reason == REASON_PERSISTENCE_WORK_OVER_CAP:
                return _refused_section(limits, REASON_PERSISTENCE_WORK_OVER_CAP,
                                        error.field)
            else:
                entry, metadata = refuse(
                    REASON_PERSISTENCE_WITNESS_UNPERSISTABLE, error.reason)
            refused = charge_refusal(entry, metadata)
            if refused is not None:
                return refused
            continue
        if total_payload + account.payload_bytes > limits.max_total_payload_bytes:
            entry, metadata = refuse(REASON_PERSISTENCE_TOTAL_PAYLOAD_OVER_CAP,
                                     str(total_payload + account.payload_bytes))
            refused = charge_refusal(entry, metadata)
            if refused is not None:
                return refused
            continue
        # Entry-level metadata includes the identity fields, the entry schema and
        # the whole witness metadata (rectangle cardinalities included); it is
        # measured from the declared cardinalities without ever building a
        # per-cell rectangle list, and before any array is copied or encoded.
        entry_metadata = _declared_entry_metadata_bytes({
            "outcome": SAMPLING_PERSISTENCE_OUTCOME_BOUND,
            "ordinal": ordinal,
            "schema": SAMPLING_PERSISTENCE_ENTRY_SCHEMA,
            **identity,
        }, item.witness)
        if entry_metadata > limits.max_entry_metadata_bytes:
            entry, metadata = refuse(REASON_PERSISTENCE_ENTRY_METADATA_OVER_CAP,
                                     str(entry_metadata))
            refused = charge_refusal(entry, metadata)
            if refused is not None:
                return refused
            continue
        if total_metadata + entry_metadata > limits.max_total_metadata_bytes:
            entry, metadata = refuse(REASON_PERSISTENCE_TOTAL_METADATA_OVER_CAP,
                                     str(total_metadata + entry_metadata))
            refused = charge_refusal(entry, metadata)
            if refused is not None:
                return refused
            continue
        entry_estimate = _entry_serialized_estimate(
            entry_metadata, account.payload_bytes)
        if estimated + entry_estimate > limits.max_serialized_bytes:
            entry, metadata = refuse(REASON_PERSISTENCE_SERIALIZED_OVER_CAP,
                                     str(estimated + entry_estimate))
            refused = charge_refusal(entry, metadata)
            if refused is not None:
                return refused
            continue
        if total_work + reserved_work + account.work_units \
                > limits.max_work_units:
            return _refused_section(limits, REASON_PERSISTENCE_WORK_OVER_CAP,
                                    "work_units")
        # --- Reserve the declared work, then scan and encode. ------------------
        # The reservation is made *before* the pointer/index/weight/bilinear value
        # scans and is never refunded: a witness whose scans refuse it
        # semantically has still consumed its declared work, so a later witness
        # cannot spend the same budget. ``_encode_witness`` materializes the
        # per-cell rectangle lists only after every applicable budget passed.
        reserved_work += account.work_units
        try:
            _validate_witness_values(item.witness, limits=limits)
            witness_record = _encode_witness(item.witness)
        except SamplingPersistenceError as error:
            entry, metadata = refuse(
                REASON_PERSISTENCE_WITNESS_UNPERSISTABLE, error.reason)
            refused = charge_refusal(entry, metadata, reserved=True)
            if refused is not None:
                return refused
            continue
        # The scans and the encoder succeeded: move the reservation into the
        # serialized accounting written to the section.
        reserved_work -= account.work_units
        total_work += account.work_units
        entry = {
            "outcome": SAMPLING_PERSISTENCE_OUTCOME_BOUND,
            "ordinal": ordinal,
            "schema": SAMPLING_PERSISTENCE_ENTRY_SCHEMA,
            "witness": witness_record,
            **identity,
        }
        entries.append(entry)
        total_payload += account.payload_bytes
        total_metadata += entry_metadata
        estimated += entry_estimate
    section = {
        "schema": SAMPLING_PERSISTENCE_SCHEMA,
        "status": SAMPLING_PERSISTENCE_STATUS_COMPLETE,
        "reason": "",
        "field": "",
        "detail": "",
        "policy": limits.to_record(),
        "entry_count": len(entries),
        "entries": entries,
        "payload_bytes": total_payload,
        "metadata_bytes": total_metadata,
        "work_units": total_work,
    }
    # The seal adds exactly one member, ``"section_sha256":"<64 hex>"`` (84
    # canonical bytes). Probe the exact sealed size with a 64-character
    # placeholder digest so an over-cap complete section is refused by the
    # declared walk before ``_seal_section``/``_canonical_bytes`` ever builds its
    # serialization; the final byte check below is only a defensive backstop.
    if _sealed_section_size(section) > limits.max_serialized_bytes:
        return _refused_section(limits, REASON_PERSISTENCE_SERIALIZED_OVER_CAP,
                                "section")
    sealed = _seal_section(section)
    if len(_canonical_bytes(sealed)) > limits.max_serialized_bytes:
        return _refused_section(limits, REASON_PERSISTENCE_SERIALIZED_OVER_CAP,
                                "section")
    return sealed


# --- Plain section validation -------------------------------------------------

@dataclass(frozen=True, slots=True)
class PersistenceAccountingV1:
    entry_count: int
    bound_count: int
    refused_count: int
    persistence_refused_count: int
    payload_bytes: int
    metadata_bytes: int
    work_units: int


_ENTRY_REQUIRED = ("outcome", "ordinal", "frame_id", "canonical_index",
                   "alternative_index")
_IDENTITY_REQUIRED = ("signature_sha256", "ingress_sha256", "mask_sha256",
                      "masks_sha256", "max_cells", "max_rectified_side",
                      "pixels_per_cell", "witness_sha256")
_BOUND_ONLY = ("schema", "witness", *_IDENTITY_REQUIRED)
_REFUSED_ONLY = ("reason", "detail")


def _validate_identity_scalars(entry: dict, field: str) -> None:
    """Primitive identity checks that must run before any dedup key or hash.

    A malformed ``frame_id`` (a list, ``None``, an empty string) or a non-integer
    ordinal must be a typed refusal, never a raw ``TypeError`` from inserting an
    unhashable key into the seen-set.
    """
    if type(entry.get("frame_id")) is not str or not entry["frame_id"]:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECORD_MALFORMED, f"{field}.frame_id")
    if not _is_index(entry.get("canonical_index")):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECORD_MALFORMED, f"{field}.canonical_index")
    if not _is_index(entry.get("alternative_index")):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECORD_MALFORMED, f"{field}.alternative_index")


def _validate_identity(entry: dict, field: str, *, required: Sequence[str]) -> None:
    for name in required:
        if name not in entry:
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_RECORD_MALFORMED, f"{field}.{name}")
    _validate_identity_scalars(entry, field)
    if not _hex64(entry.get("signature_sha256")) \
            or not _hex64(entry.get("ingress_sha256")) \
            or not _hex64(entry.get("mask_sha256")) \
            or not _hex64(entry.get("masks_sha256")) \
            or not _hex64(entry.get("witness_sha256")):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_HASH_MISMATCH, field)
    for name in ("max_cells", "max_rectified_side", "pixels_per_cell"):
        if not _is_positive_int(entry.get(name)):
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_RECORD_MALFORMED, f"{field}.{name}")


def _validate_witness_plain(witness: object, *,
                            entry_field: str,
                            limits: SamplingPersistenceLimitsV1,
                            remaining: int) -> WitnessAccountingV1:
    witness = _as_dict(witness, f"{entry_field}.witness")
    _require_keys(
        witness,
        ("schema", "grayscale_sha256", "gradient_y_sha256", "gradient_x_sha256",
         "energy_sha256", "coverage_patch_sha256", "gradient_edge_order",
         "subsample_factor", "seam_band", "cell_rectangles",
         "interior_rectangles", "rgb_rectification",
         "coverage_rectification", "region_rectification", "cell_source_ptr",
         "cell_source_indices", "witness_sha256"),
        (), f"{entry_field}.witness")
    if witness["schema"] != "mc-photogrammetry-signature-sampling-witness-v1":
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_SCHEMA_INVALID, f"{entry_field}.witness.schema")
    if type(witness["gradient_edge_order"]) is not int \
            or witness["gradient_edge_order"] != 1:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_SCHEMA_INVALID,
            f"{entry_field}.witness.gradient_edge_order")
    if (not _is_positive_int(witness["subsample_factor"])
            or not _is_positive_int(witness["seam_band"])):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_SCHEMA_INVALID,
            f"{entry_field}.witness.sampling_settings")
    for name in ("grayscale_sha256", "gradient_y_sha256", "gradient_x_sha256",
                 "energy_sha256", "coverage_patch_sha256", "witness_sha256"):
        if not _hex64(witness[name]):
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_HASH_MISMATCH,
                f"{entry_field}.witness.{name}")
    rectangles = witness["cell_rectangles"]
    interiors = witness["interior_rectangles"]
    # Interior rectangles are type-validated *before* their length is compared,
    # so ``interior_rectangles=null`` is a typed rectangle refusal rather than a
    # raw ``TypeError`` from ``len(None)``.
    _validate_rectangles(rectangles, field=f"{entry_field}.witness.rectangles",
                         limits=limits)
    _validate_rectangles(interiors,
                         field=f"{entry_field}.witness.interior_rectangles",
                         limits=limits)
    if len(interiors) != len(rectangles):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECTANGLE_INVALID,
            f"{entry_field}.witness.interior_rectangles")
    maximum_array_size = limits.max_rectified_side * limits.max_rectified_side
    total = 0
    elements = 0
    rectification_count = 0
    pointer_shape, pointer_size, pointer_bytes = _declared_array_bytes(
        witness["cell_source_ptr"], field="cell_source_ptr",
        remaining=remaining, max_size=maximum_array_size)
    if pointer_size != len(rectangles) + 1:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_INDEX_INVALID,
            f"{entry_field}.witness.cell_source_ptr")
    index_shape, index_size, index_bytes = _declared_array_bytes(
        witness["cell_source_indices"], field="cell_source_indices",
        remaining=remaining - pointer_bytes, max_size=maximum_array_size)
    total += pointer_bytes + index_bytes
    elements += pointer_size + index_size
    for field, required in (("rgb_rectification", True),
                            ("coverage_rectification", True),
                            ("region_rectification", False)):
        rectification = witness[field]
        if rectification is None:
            if required:
                raise SamplingPersistenceError(
                    REASON_PERSISTENCE_RECORD_MALFORMED,
                    f"{entry_field}.witness.{field}")
            continue
        nbytes, _rect_array_count = _validate_rectification_plain(
            rectification, field=f"{entry_field}.witness.{field}",
            limits=limits, remaining=remaining - total)
        total += nbytes
        rectification_count += 1
    metadata_bytes = canonical_json_size(_plain_witness_metadata(witness))
    work = _declared_witness_work(
        total, metadata_bytes, len(rectangles), len(interiors),
        2 + len(_RECTIFICATION_ARRAY_FIELDS) * rectification_count)
    return WitnessAccountingV1(
        payload_bytes=total, payload_elements=elements,
        metadata_bytes=metadata_bytes, work_units=work)


def _plain_witness_metadata(witness: dict) -> dict:
    """The witness record with every array's ``b64`` payload removed."""
    metadata = dict(witness)
    for name in ("cell_source_ptr", "cell_source_indices"):
        metadata[name] = _strip_b64(witness[name])
    for field in ("rgb_rectification", "coverage_rectification",
                  "region_rectification"):
        rectification = witness[field]
        if rectification is None:
            continue
        rect_meta = dict(rectification)
        for name in _RECTIFICATION_ARRAY_FIELDS:
            rect_meta[name] = _strip_b64(rectification[name])
        metadata[field] = rect_meta
    return metadata


def _strip_b64(record: object) -> dict:
    record = _as_dict(record, "array")
    return {key: value for key, value in record.items() if key != "b64"}


def _validate_rectification_plain(rectification: object, *, field: str,
                                  limits: SamplingPersistenceLimitsV1,
                                  remaining: int) -> tuple[int, int]:
    record = _as_dict(rectification, field)
    _require_keys(
        record,
        ("schema", "opencv_version", "opencv_build_sha256", "interpolation",
         "border", "inter_bits", "weight_denominator", "source_shape",
         "witness_sha256", *_RECTIFICATION_ARRAY_FIELDS),
        (), field)
    if record["schema"] != "mc-photogrammetry-rectification-witness-v1":
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_SCHEMA_INVALID, f"{field}.schema")
    if record["interpolation"] != "INTER_LINEAR" \
            or record["border"] != "BORDER_CONSTANT_ZERO" \
            or type(record["inter_bits"]) is not int or record["inter_bits"] != 5 \
            or type(record["weight_denominator"]) is not int \
            or record["weight_denominator"] != 1024:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_SCHEMA_INVALID, f"{field}.backend")
    if not _hex64(record["witness_sha256"]) \
            or not _hex64(record["opencv_build_sha256"]):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_HASH_MISMATCH, f"{field}.witness_sha256")
    if type(record["opencv_version"]) is not str or not record["opencv_version"]:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_SCHEMA_INVALID, f"{field}.opencv_version")
    source_shape = record["source_shape"]
    if (type(source_shape) is not list or len(source_shape) != 2
            or any(not _is_positive_int(value) for value in source_shape)
            or any(value > limits.max_source_side for value in source_shape)
            or _product(source_shape) > limits.max_source_pixels):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECTANGLE_INVALID, f"{field}.source_shape")
    maximum_array_size = limits.max_rectified_side * limits.max_rectified_side
    total = 0
    map_shape = None
    for name in _RECTIFICATION_ARRAY_FIELDS:
        shape, size, nbytes = _declared_array_bytes(
            record[name], field=name, remaining=remaining - total,
            max_size=maximum_array_size)
        if name == "map_x":
            map_shape = shape
            if (len(shape) != 2 or any(value <= 0
                                       or value > limits.max_rectified_side
                                       for value in shape)):
                raise SamplingPersistenceError(
                    REASON_PERSISTENCE_RECTANGLE_INVALID, f"{field}.map_x.shape")
        elif name in ("map_y", "depth_valid") and shape != map_shape:
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_RECTANGLE_INVALID, f"{field}.{name}.shape")
        elif name == "tap_ptr" and map_shape is not None \
                and size != _product(map_shape) + 1:
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_INDEX_INVALID, f"{field}.tap_ptr.shape")
        total += nbytes
    return total, 1 + len(_RECTIFICATION_ARRAY_FIELDS)


def validate_section_plain(
    section: object,
    *,
    limits: SamplingPersistenceLimitsV1 = SAMPLING_PERSISTENCE_LIMITS,
) -> PersistenceAccountingV1:
    """Cap, structure and digest validation on the plain parsed section.

    No base64 is decoded and no NumPy array is constructed here. Scalar types,
    entry count/order, encoded lengths, the per-entry and aggregate declared
    budgets (payload, metadata and work) and the refusal identity/detail sizes are
    all validated first; the section digest is recomputed and compared **last**,
    so no over-cap section is ever canonically serialized just to hash it.
    """
    section = _as_dict(section, "sampling_persistence")
    policy = SamplingPersistenceLimitsV1.from_record(section.get("policy"))
    if policy.to_record() != limits.to_record():
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_POLICY_MISMATCH, "policy")
    if section.get("schema") != SAMPLING_PERSISTENCE_SCHEMA:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_SCHEMA_INVALID, "section.schema")
    status = section.get("status")
    if status not in (SAMPLING_PERSISTENCE_STATUS_COMPLETE,
                      SAMPLING_PERSISTENCE_STATUS_REFUSED):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECORD_MALFORMED, "section.status")
    if _document_depth(section, cap=limits.max_depth) > limits.max_depth:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_DEPTH_OVER_CAP, "section")
    if status == SAMPLING_PERSISTENCE_STATUS_REFUSED:
        _require_keys(section,
                      ("schema", "status", "reason", "field", "detail",
                       "policy", "entry_count", "entries", "payload_bytes",
                       "metadata_bytes", "work_units", "section_sha256"), (),
                      "section")
        if (type(section["reason"]) is not str
                or section["reason"] not in PERSISTENCE_REFUSAL_REASONS
                or type(section["field"]) is not str
                or type(section["detail"]) is not str
                or len(section["field"]) + len(section["detail"])
                > limits.max_entry_metadata_bytes
                or type(section["entry_count"]) is not int
                or section["entry_count"] != 0 or section["entries"] != []
                or type(section["payload_bytes"]) is not int
                or section["payload_bytes"] != 0
                or type(section["metadata_bytes"]) is not int
                or section["metadata_bytes"] != 0
                or type(section["work_units"]) is not int
                or section["work_units"] != 0):
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_RECORD_MALFORMED, "section.refused")
        _check_section_serialized_cap(section, limits)
        _verify_section_digest(section)
        return PersistenceAccountingV1(
            entry_count=0, bound_count=0, refused_count=0,
            persistence_refused_count=0, payload_bytes=0, metadata_bytes=0,
            work_units=0)
    _require_keys(section,
                  ("schema", "status", "reason", "field", "detail", "policy",
                   "entry_count", "entries", "payload_bytes", "metadata_bytes",
                   "work_units", "section_sha256"), (), "section")
    if (type(section["reason"]) is not str or type(section["field"]) is not str
            or type(section["detail"]) is not str
            or section["reason"] or section["field"] or section["detail"]):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECORD_MALFORMED, "section.reason")
    entries = _as_list(section["entries"], "section.entries")
    if type(section["entry_count"]) is not int \
            or section["entry_count"] != len(entries):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECORD_MALFORMED, "section.entry_count")
    if len(entries) > limits.max_entries:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_ENTRY_CAP, "section.entries")
    total_payload = 0
    total_metadata = 0
    total_work = 0
    bound = 0
    refused = 0
    persistence_refused = 0
    seen: set[tuple] = set()
    for ordinal, entry in enumerate(entries):
        entry = _as_dict(entry, f"entries.{ordinal}")
        outcome = entry.get("outcome")
        if type(outcome) is not str or outcome not in (
                SAMPLING_PERSISTENCE_OUTCOME_BOUND,
                SAMPLING_PERSISTENCE_OUTCOME_REFUSED,
                SAMPLING_PERSISTENCE_OUTCOME_PERSISTENCE_REFUSED):
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_OUTCOME_INVALID, f"entries.{ordinal}.outcome")
        _require_keys(entry, _ENTRY_REQUIRED, (*_BOUND_ONLY, *_REFUSED_ONLY),
                      f"entries.{ordinal}")
        # The ordinal must be an exact integer *before* the equality test: a
        # float ``0.0`` (or ``True``) must never satisfy ordinal ``0``.
        if type(entry["ordinal"]) is not int or entry["ordinal"] != ordinal:
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_RECORD_MALFORMED, f"entries.{ordinal}.ordinal")
        # Primitive identity validation before the dedup set: an unhashable
        # malformed identity must be a typed refusal, not a raw TypeError.
        _validate_identity_scalars(entry, f"entries.{ordinal}")
        identity_key = (entry["frame_id"], entry["canonical_index"],
                        entry["alternative_index"])
        if identity_key in seen:
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_DUPLICATE_ENTRY, f"entries.{ordinal}")
        seen.add(identity_key)
        if outcome == SAMPLING_PERSISTENCE_OUTCOME_BOUND:
            _require_keys(entry, (*_ENTRY_REQUIRED, *_BOUND_ONLY),
                          _REFUSED_ONLY, f"entries.{ordinal}")
            _validate_identity(entry, f"entries.{ordinal}",
                               required=_IDENTITY_REQUIRED)
            if entry["schema"] != SAMPLING_PERSISTENCE_ENTRY_SCHEMA:
                raise SamplingPersistenceError(
                    REASON_PERSISTENCE_SCHEMA_INVALID,
                    f"entries.{ordinal}.schema")
            account = _validate_witness_plain(
                entry["witness"], entry_field=f"entries.{ordinal}",
                limits=limits,
                remaining=limits.max_total_payload_bytes - total_payload)
            if account.payload_bytes > limits.max_entry_payload_bytes:
                raise SamplingPersistenceError(
                    REASON_PERSISTENCE_ENTRY_PAYLOAD_OVER_CAP,
                    f"entries.{ordinal}")
            if total_payload + account.payload_bytes \
                    > limits.max_total_payload_bytes:
                raise SamplingPersistenceError(
                    REASON_PERSISTENCE_TOTAL_PAYLOAD_OVER_CAP, "payload_bytes")
            entry_metadata = _entry_metadata_bytes(
                entry, witness_metadata=_plain_witness_metadata(entry["witness"]))
            if entry_metadata > limits.max_entry_metadata_bytes:
                raise SamplingPersistenceError(
                    REASON_PERSISTENCE_ENTRY_METADATA_OVER_CAP,
                    f"entries.{ordinal}.metadata")
            if total_metadata + entry_metadata \
                    > limits.max_total_metadata_bytes:
                raise SamplingPersistenceError(
                    REASON_PERSISTENCE_TOTAL_METADATA_OVER_CAP, "metadata_bytes")
            total_payload += account.payload_bytes
            total_metadata += entry_metadata
            total_work += account.work_units
            bound += 1
        elif outcome == SAMPLING_PERSISTENCE_OUTCOME_REFUSED:
            _require_keys(entry, (*_ENTRY_REQUIRED, *_REFUSED_ONLY),
                          _BOUND_ONLY, f"entries.{ordinal}")
            if type(entry["reason"]) is not str or entry["reason"] not in (
                    provenance.SOURCE_TAP_WITNESS_UNAVAILABLE,
                    provenance.SOURCE_TAP_WITNESS_BUDGET_EXHAUSTED,
                    provenance.SOURCE_TAP_WITNESS_SHAPE_MISMATCH,
                    provenance.SOURCE_TAP_WITNESS_INDEX_OUT_OF_RANGE,
                    provenance.INGRESS_PROVENANCE_UNAVAILABLE):
                raise SamplingPersistenceError(
                    REASON_PERSISTENCE_OUTCOME_INVALID,
                    f"entries.{ordinal}.reason")
            refused_metadata = _non_bound_entry_metadata_bytes(
                entry, limit=limits.max_entry_metadata_bytes)
            # The exact escaped canonical JSON size of the whole refusal entry
            # (identity, ordinal, outcome, reason and detail) must fit the
            # per-entry metadata cap as well as the aggregate cap.
            if refused_metadata > limits.max_entry_metadata_bytes:
                raise SamplingPersistenceError(
                    REASON_PERSISTENCE_ENTRY_METADATA_OVER_CAP,
                    f"entries.{ordinal}.metadata")
            if total_metadata + refused_metadata \
                    > limits.max_total_metadata_bytes:
                raise SamplingPersistenceError(
                    REASON_PERSISTENCE_TOTAL_METADATA_OVER_CAP, "metadata_bytes")
            total_metadata += refused_metadata
            # A non-bound outcome charges one declared work unit *plus* its exact
            # metadata bytes (the metadata is traversed and JSON-encoded); this is
            # the identical charge ``persist_signature_witnesses`` applies, so a
            # large refusal detail cannot validate complete under a work cap it
            # exceeds.
            total_work += 1 + refused_metadata
            refused += 1
        else:
            # A persistence refusal may carry the full binding identity (a bound
            # witness degraded by a budget) or only the refusal identity (an
            # over-cap refusal detail degraded to a small typed record); a partial
            # identity is malformed either way.
            present = [name for name in _IDENTITY_REQUIRED if name in entry]
            if present and len(present) != len(_IDENTITY_REQUIRED):
                raise SamplingPersistenceError(
                    REASON_PERSISTENCE_RECORD_MALFORMED,
                    f"entries.{ordinal}.identity")
            _require_keys(
                entry,
                (*_ENTRY_REQUIRED, *_REFUSED_ONLY),
                (*_IDENTITY_REQUIRED,), f"entries.{ordinal}")
            if type(entry["reason"]) is not str \
                    or entry["reason"] not in PERSISTENCE_REFUSAL_REASONS:
                raise SamplingPersistenceError(
                    REASON_PERSISTENCE_OUTCOME_INVALID,
                    f"entries.{ordinal}.reason")
            if present:
                _validate_identity(entry, f"entries.{ordinal}",
                                   required=_IDENTITY_REQUIRED)
            else:
                _validate_identity_scalars(entry, f"entries.{ordinal}")
            refused_metadata = _non_bound_entry_metadata_bytes(
                entry, limit=limits.max_entry_metadata_bytes)
            if refused_metadata > limits.max_entry_metadata_bytes:
                raise SamplingPersistenceError(
                    REASON_PERSISTENCE_ENTRY_METADATA_OVER_CAP,
                    f"entries.{ordinal}.metadata")
            if total_metadata + refused_metadata \
                    > limits.max_total_metadata_bytes:
                raise SamplingPersistenceError(
                    REASON_PERSISTENCE_TOTAL_METADATA_OVER_CAP, "metadata_bytes")
            total_metadata += refused_metadata
            total_work += 1 + refused_metadata
            persistence_refused += 1
        if total_work > limits.max_work_units:
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_WORK_OVER_CAP, "work_units")
    if (type(section["payload_bytes"]) is not int
            or type(section["metadata_bytes"]) is not int
            or type(section["work_units"]) is not int):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECORD_MALFORMED, "section.accounting")
    if (section["payload_bytes"] != total_payload
            or section["metadata_bytes"] != total_metadata
            or section["work_units"] != total_work):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_DIGEST_MISMATCH, "section.accounting")
    # The complete section's own declared serialized size is capped *before* it is
    # canonically serialized to recompute and compare its digest.
    _check_section_serialized_cap(section, limits)
    # Only now, after every scalar, type, length, declared budget and the section
    # size cap above passed, is the section serialized to verify its digest.
    _verify_section_digest(section)
    return PersistenceAccountingV1(
        entry_count=len(entries), bound_count=bound, refused_count=refused,
        persistence_refused_count=persistence_refused,
        payload_bytes=total_payload, metadata_bytes=total_metadata,
        work_units=total_work)


def _check_section_serialized_cap(section: dict,
                                  limits: SamplingPersistenceLimitsV1) -> None:
    """Refuse a section whose exact declared serialized size exceeds the cap.

    ``canonical_json_size`` walks the plain JSON structure and returns the exact
    canonical byte length **without building the string**, so an over-cap section
    is refused before ``section_digest``/``_canonical_bytes`` ever serializes it.
    """
    if canonical_json_size(section) > limits.max_serialized_bytes:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_SERIALIZED_OVER_CAP, "section")


def _verify_section_digest(section: dict) -> None:
    claimed = section.get("section_sha256")
    if not _hex64(claimed) or claimed != section_digest(section):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_DIGEST_MISMATCH, "section_sha256")


# --- Typed decode -------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SamplingPersistenceEntryV1:
    ordinal: int
    outcome: str
    frame_id: str
    canonical_index: int
    alternative_index: int
    signature_sha256: str = ""
    ingress_sha256: str = ""
    mask_sha256: str = ""
    masks_sha256: str = ""
    max_cells: int = 0
    max_rectified_side: int = 0
    pixels_per_cell: int = 0
    witness_sha256: str = ""
    reason: str = ""
    detail: str = ""
    witness: lattice_planes.SignatureSamplingWitnessV1 | None = None
    binding: provenance.SourceTapWitnessBindingV1 | None = None

    @property
    def has_payload(self) -> bool:
        return self.witness is not None


@dataclass(frozen=True, slots=True)
class SignatureSamplingPersistenceSectionV1:
    status: str
    reason: str
    policy: SamplingPersistenceLimitsV1
    entries: tuple[SamplingPersistenceEntryV1, ...]
    payload_bytes: int
    metadata_bytes: int
    work_units: int
    section_sha256: str
    schema: str = SAMPLING_PERSISTENCE_SCHEMA

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    @property
    def bound_count(self) -> int:
        return sum(1 for entry in self.entries
                   if entry.outcome == SAMPLING_PERSISTENCE_OUTCOME_BOUND)

    @property
    def refused_count(self) -> int:
        return sum(1 for entry in self.entries
                   if entry.outcome == SAMPLING_PERSISTENCE_OUTCOME_REFUSED)

    @property
    def persistence_refused_count(self) -> int:
        return sum(1 for entry in self.entries
                   if entry.outcome
                   == SAMPLING_PERSISTENCE_OUTCOME_PERSISTENCE_REFUSED)

    @property
    def state(self) -> str:
        """``available`` only when every bound payload decoded against a root pin.

        A complete section with no persistence refusal but no ingress binding is
        *not* available: the source-space witness can only be trusted once the
        record's own capture-root ingress pins were supplied and matched, so an
        unpinned (or foreign-root) bound witness always stays unavailable.
        """
        if (self.status == SAMPLING_PERSISTENCE_STATUS_COMPLETE
                and self.persistence_refused_count == 0
                and self.bound_count > 0
                and all(entry.binding is not None for entry in self.entries
                        if entry.outcome == SAMPLING_PERSISTENCE_OUTCOME_BOUND)):
            return SAMPLING_PERSISTENCE_STATE_AVAILABLE
        return SAMPLING_PERSISTENCE_STATE_UNAVAILABLE


def _decode_rectification(record: object, *, field: str, limits,
                          remaining: int) -> tuple[lattice.RectificationWitnessV1,
                                                   int]:
    _validate_rectification_plain(record, field=field, limits=limits,
                                  remaining=remaining)
    record = _as_dict(record, field)
    arrays = {}
    total = 0
    map_size = None
    for name in _RECTIFICATION_ARRAY_FIELDS:
        array = _decode_array_record(
            record[name], field=name,
            remaining=limits.max_entry_payload_bytes - total,
            max_size=limits.max_rectified_side * limits.max_rectified_side)
        if name == "map_x":
            map_size = int(array.size)
        total += int(array.nbytes)
        arrays[name] = array
    if map_size is None:  # pragma: no cover - validated above
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_RECORD_MALFORMED, field)
    _validate_pointer_array(arrays["tap_ptr"], field=f"{field}.tap_ptr",
                            final_length=int(arrays["tap_source_indices"].size))
    _validate_index_array(
        arrays["tap_source_indices"], field=f"{field}.tap_source_indices",
        source_cells=int(record["source_shape"][0]) * int(record["source_shape"][1]))
    _validate_weight_array(
        arrays["tap_weight_numerators"], field=f"{field}.tap_weight_numerators",
        denominator=record["weight_denominator"])
    _validate_bilinear_tap_semantics(
        samples=int(arrays["map_x"].size), depth_valid=arrays["depth_valid"],
        tap_ptr=arrays["tap_ptr"], indices=arrays["tap_source_indices"],
        weights=arrays["tap_weight_numerators"], field=field,
        denominator=record["weight_denominator"])
    try:
        witness = lattice.RectificationWitnessV1(
            schema=record["schema"], opencv_version=record["opencv_version"],
            opencv_build_sha256=record["opencv_build_sha256"],
            interpolation=record["interpolation"], border=record["border"],
            inter_bits=record["inter_bits"],
            weight_denominator=record["weight_denominator"],
            source_shape=tuple(record["source_shape"]),
            map_x=arrays["map_x"], map_y=arrays["map_y"],
            depth_valid=arrays["depth_valid"], tap_ptr=arrays["tap_ptr"],
            tap_source_indices=arrays["tap_source_indices"],
            tap_weight_numerators=arrays["tap_weight_numerators"],
            witness_sha256=record["witness_sha256"])
    except (TypeError, ValueError, OverflowError) as error:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_SCHEMA_INVALID, field,
            type(error).__name__) from error
    return witness, total


def _decode_witness(entry: dict, *, field: str, limits) -> tuple[
        lattice_planes.SignatureSamplingWitnessV1, int]:
    account = _validate_witness_plain(
        entry["witness"], entry_field=field, limits=limits,
        remaining=limits.max_total_payload_bytes)
    witness_record = _as_dict(entry["witness"], f"{field}.witness")
    total = 0
    pointer = _decode_array_record(
        witness_record["cell_source_ptr"], field="cell_source_ptr",
        remaining=limits.max_entry_payload_bytes - total,
        max_size=limits.max_rectified_side * limits.max_rectified_side)
    total += int(pointer.nbytes)
    indices = _decode_array_record(
        witness_record["cell_source_indices"], field="cell_source_indices",
        remaining=limits.max_entry_payload_bytes - total,
        max_size=limits.max_rectified_side * limits.max_rectified_side)
    total += int(indices.nbytes)
    _validate_pointer_array(pointer, field=f"{field}.cell_source_ptr",
                            final_length=int(indices.size))
    source_cells = 1
    rectifications: dict[str, lattice.RectificationWitnessV1 | None] = {}
    for name in ("rgb_rectification", "coverage_rectification",
                 "region_rectification"):
        record = witness_record[name]
        if record is None:
            if name != "region_rectification":
                raise SamplingPersistenceError(
                    REASON_PERSISTENCE_RECORD_MALFORMED, f"{field}.{name}")
            rectifications[name] = None
            continue
        witness, nbytes = _decode_rectification(
            record, field=f"{field}.{name}", limits=limits,
            remaining=limits.max_entry_payload_bytes - total)
        total += nbytes
        rectifications[name] = witness
        source_cells = max(source_cells,
                           int(witness.source_shape[0]) * int(witness.source_shape[1]))
    _validate_index_array(indices, field=f"{field}.cell_source_indices",
                          source_cells=source_cells)
    try:
        witness = lattice_planes.SignatureSamplingWitnessV1(
            schema=witness_record["schema"],
            rgb_rectification=rectifications["rgb_rectification"],
            coverage_rectification=rectifications["coverage_rectification"],
            region_rectification=rectifications["region_rectification"],
            grayscale_sha256=witness_record["grayscale_sha256"],
            gradient_y_sha256=witness_record["gradient_y_sha256"],
            gradient_x_sha256=witness_record["gradient_x_sha256"],
            energy_sha256=witness_record["energy_sha256"],
            coverage_patch_sha256=witness_record["coverage_patch_sha256"],
            gradient_edge_order=witness_record["gradient_edge_order"],
            subsample_factor=witness_record["subsample_factor"],
            seam_band=witness_record["seam_band"],
            cell_rectangles=tuple(tuple(value) for value
                                  in witness_record["cell_rectangles"]),
            interior_rectangles=tuple(tuple(value) for value
                                      in witness_record["interior_rectangles"]),
            cell_source_ptr=pointer, cell_source_indices=indices,
            witness_sha256=witness_record["witness_sha256"])
    except (TypeError, ValueError, OverflowError) as error:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_SCHEMA_INVALID, field,
            type(error).__name__) from error
    if witness.witness_sha256 != entry["witness_sha256"]:
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_DIGEST_MISMATCH, f"{field}.witness_sha256")
    return witness, total


def decode_section(
    section: object,
    *,
    limits: SamplingPersistenceLimitsV1 = SAMPLING_PERSISTENCE_LIMITS,
    ingress_by_frame: dict | None = None,
    require_ingress: bool = True,
) -> SignatureSamplingPersistenceSectionV1:
    """Validate and decode one section into immutable typed outcomes.

    ``validate_section_plain`` runs first, so every scalar type, entry order,
    encoded length and declared payload/metadata/work budget has already been
    checked before a single base64 string is decoded or a NumPy array is
    allocated. A bound entry is additionally rebuilt as a
    :class:`~...candidate_support_provenance.SourceTapWitnessBindingV1` from its
    root-pinned ingress: the pin's ``ingress_sha256`` must match exactly and the
    binding constructor re-checks every rectification ``source_shape`` against the
    pinned frame shape. A bound entry whose frame has no supplied pin refuses when
    ``require_ingress`` is true (the default) and stays unbound - therefore never
    ``available`` - otherwise.
    """
    account = validate_section_plain(section, limits=limits)
    section = _as_dict(section, "sampling_persistence")
    policy = SamplingPersistenceLimitsV1.from_record(section["policy"])
    if section["status"] == SAMPLING_PERSISTENCE_STATUS_REFUSED:
        return SignatureSamplingPersistenceSectionV1(
            status=section["status"], reason=section["reason"], policy=policy,
            entries=(), payload_bytes=0, metadata_bytes=0, work_units=0,
            section_sha256=section["section_sha256"])
    entries: list[SamplingPersistenceEntryV1] = []
    for ordinal, entry in enumerate(section["entries"]):
        entry = _as_dict(entry, f"entries.{ordinal}")
        outcome = entry["outcome"]
        if outcome != SAMPLING_PERSISTENCE_OUTCOME_BOUND:
            entries.append(SamplingPersistenceEntryV1(
                ordinal=ordinal, outcome=outcome, frame_id=entry["frame_id"],
                canonical_index=entry["canonical_index"],
                alternative_index=entry["alternative_index"],
                signature_sha256=entry.get("signature_sha256", ""),
                ingress_sha256=entry.get("ingress_sha256", ""),
                mask_sha256=entry.get("mask_sha256", ""),
                masks_sha256=entry.get("masks_sha256", ""),
                max_cells=entry.get("max_cells", 0),
                max_rectified_side=entry.get("max_rectified_side", 0),
                pixels_per_cell=entry.get("pixels_per_cell", 0),
                witness_sha256=entry.get("witness_sha256", ""),
                reason=entry.get("reason", ""), detail=entry.get("detail", "")))
            continue
        witness, _ = _decode_witness(
            entry, field=f"entries.{ordinal}", limits=limits)
        binding = None
        pins = {} if ingress_by_frame is None else ingress_by_frame
        ingress = pins.get(entry["frame_id"])
        if ingress is None:
            # No root pin for this frame: a bound witness can never be reported
            # as available from it. With ``require_ingress`` (the default) this is
            # a typed refusal rather than a silently unbound outcome.
            if require_ingress:
                raise SamplingPersistenceError(
                    REASON_PERSISTENCE_INGRESS_MISMATCH,
                    f"entries.{ordinal}.ingress")
        elif not isinstance(ingress, provenance.FrameIngressProvenanceV1):
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_INGRESS_MISMATCH,
                f"entries.{ordinal}.ingress")
        elif _digest(ingress.to_record()) != entry["ingress_sha256"]:
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_INGRESS_MISMATCH,
                f"entries.{ordinal}.ingress_sha256")
        else:
            try:
                binding = provenance.SourceTapWitnessBindingV1(
                    frame_id=entry["frame_id"],
                    canonical_index=entry["canonical_index"],
                    alternative_index=entry["alternative_index"],
                    signature_sha256=entry["signature_sha256"], ingress=ingress,
                    mask_sha256=entry["mask_sha256"],
                    masks_sha256=entry["masks_sha256"],
                    max_cells=entry["max_cells"],
                    max_rectified_side=entry["max_rectified_side"],
                    pixels_per_cell=entry["pixels_per_cell"], witness=witness,
                    witness_sha256=entry["witness_sha256"])
            except (TypeError, ValueError, OverflowError) as error:
                # ``SourceTapWitnessBindingV1`` re-checks the rectification
                # ``source_shape`` against the pinned ingress shape here, so a
                # witness whose source shape is not the pinned root frame is
                # refused rather than rebound.
                raise SamplingPersistenceError(
                    REASON_PERSISTENCE_INGRESS_MISMATCH,
                    f"entries.{ordinal}.source_shape",
                    type(error).__name__) from error
        entries.append(SamplingPersistenceEntryV1(
            ordinal=ordinal, outcome=outcome, frame_id=entry["frame_id"],
            canonical_index=entry["canonical_index"],
            alternative_index=entry["alternative_index"],
            signature_sha256=entry["signature_sha256"],
            ingress_sha256=entry["ingress_sha256"],
            mask_sha256=entry["mask_sha256"], masks_sha256=entry["masks_sha256"],
            max_cells=entry["max_cells"],
            max_rectified_side=entry["max_rectified_side"],
            pixels_per_cell=entry["pixels_per_cell"],
            witness_sha256=entry["witness_sha256"], witness=witness,
            binding=binding))
    return SignatureSamplingPersistenceSectionV1(
        status=section["status"], reason=section["reason"], policy=policy,
        entries=tuple(entries), payload_bytes=account.payload_bytes,
        metadata_bytes=account.metadata_bytes, work_units=account.work_units,
        section_sha256=section["section_sha256"])


# --- Cross-binding against the record's ordered witness summary ---------------

def cross_bind_witness_summaries(
    section: object,
    summaries: object,
    *,
    limits: SamplingPersistenceLimitsV1 = SAMPLING_PERSISTENCE_LIMITS,
) -> None:
    """Require the ordered section outcomes to match the record's summary rows.

    The summary rows are the runner's canonical ``signature_witnesses`` records.
    A foreign or swapped identity, a missing or duplicated entry, a modified
    array whose typed digest no longer matches the summary's ``witness_sha256``
    or a ``persistence_refused``/``refused`` outcome that disagrees with the
    original bound/refused disposition is a typed refusal.
    """
    section = _as_dict(section, "sampling_persistence")
    validate_section_plain(section, limits=limits)
    if section["status"] == SAMPLING_PERSISTENCE_STATUS_REFUSED:
        # A refused section persists no ordered outcomes; there is nothing to
        # cross-bind and the caller must treat sampling as unavailable.
        return
    summaries = _as_list(summaries, "signature_witnesses")
    entries = section["entries"]
    if len(entries) != len(summaries):
        raise SamplingPersistenceError(
            REASON_PERSISTENCE_SUMMARY_MISMATCH, "entry_count")
    for ordinal, (entry, summary) in enumerate(zip(entries, summaries)):
        entry = _as_dict(entry, f"entries.{ordinal}")
        summary = _as_dict(summary, f"signature_witnesses.{ordinal}")
        for name in ("frame_id", "canonical_index", "alternative_index"):
            if entry.get(name) != summary.get(name):
                raise SamplingPersistenceError(
                    REASON_PERSISTENCE_SUMMARY_MISMATCH,
                    f"entries.{ordinal}.{name}")
        summary_outcome = summary.get("outcome")
        if summary_outcome == "bound":
            if entry.get("outcome") not in (
                    SAMPLING_PERSISTENCE_OUTCOME_BOUND,
                    SAMPLING_PERSISTENCE_OUTCOME_PERSISTENCE_REFUSED):
                raise SamplingPersistenceError(
                    REASON_PERSISTENCE_SUMMARY_MISMATCH,
                    f"entries.{ordinal}.outcome")
            for name in ("signature_sha256", "ingress_sha256", "mask_sha256",
                         "masks_sha256", "max_cells", "max_rectified_side",
                         "pixels_per_cell", "witness_sha256"):
                if entry.get(name) != summary.get(name):
                    raise SamplingPersistenceError(
                        REASON_PERSISTENCE_SUMMARY_MISMATCH,
                        f"entries.{ordinal}.{name}")
        elif summary_outcome == "refused":
            # The original refusal is preserved in the record's ordered summary.
            # The section either carries the same refusal verbatim or - when its
            # identity/reason/detail metadata exceeded the metadata budget - a
            # small typed ``persistence_refused`` record for the same identity.
            # Nothing else is accepted.
            if entry.get("outcome") == SAMPLING_PERSISTENCE_OUTCOME_REFUSED:
                if entry.get("reason") != summary.get("reason") \
                        or entry.get("detail", "") != summary.get("detail", ""):
                    raise SamplingPersistenceError(
                        REASON_PERSISTENCE_SUMMARY_MISMATCH,
                        f"entries.{ordinal}.refusal")
            elif entry.get("outcome") \
                    != SAMPLING_PERSISTENCE_OUTCOME_PERSISTENCE_REFUSED:
                raise SamplingPersistenceError(
                    REASON_PERSISTENCE_SUMMARY_MISMATCH,
                    f"entries.{ordinal}.outcome")
        else:
            raise SamplingPersistenceError(
                REASON_PERSISTENCE_SUMMARY_MISMATCH,
                f"signature_witnesses.{ordinal}.outcome")


def section_state(section: object, *,
                  limits: SamplingPersistenceLimitsV1 = SAMPLING_PERSISTENCE_LIMITS,
                  ingress_by_frame: dict | None = None,
                  bound_summary_count: int | None = None) -> str:
    """Return ``available`` only for a fully decoded, fully bound section."""
    try:
        decoded = decode_section(section, limits=limits,
                                 ingress_by_frame=ingress_by_frame)
    except SamplingPersistenceError:
        return SAMPLING_PERSISTENCE_STATE_UNAVAILABLE
    if decoded.state != SAMPLING_PERSISTENCE_STATE_AVAILABLE:
        return SAMPLING_PERSISTENCE_STATE_UNAVAILABLE
    if bound_summary_count is not None \
            and decoded.bound_count != bound_summary_count:
        return SAMPLING_PERSISTENCE_STATE_UNAVAILABLE
    if ingress_by_frame is not None and any(
            entry.binding is None for entry in decoded.entries
            if entry.outcome == SAMPLING_PERSISTENCE_OUTCOME_BOUND):
        return SAMPLING_PERSISTENCE_STATE_UNAVAILABLE
    return SAMPLING_PERSISTENCE_STATE_AVAILABLE
