"""Evaluation-only V2 saved-support / ownership-alignment preflight.

Task ``DS-IMG-V2-SUPPORT-PREFLIGHT`` (AGENT_COMMS 2026-10-05). This module is a
bounded, fail-closed, **read-only** preflight over the ONE accepted frozen V2
Route B diagnostic record and nothing else:

* ``.tmp-routeb-v2-frozen-r4-20261004/record.json`` (the named V2 index and
  support sidecar sections are embedded in that one document as
  ``capture.index_pages`` / ``capture.sidecar_pages`` / ``capture.digest_pages``);
* no evaluator, truth, camera manifest, mask, image, scene, world, Litematic,
  JAR, network or client input is ever opened.

What it proves, from the saved record alone:

* the frozen input byte/digest pins and every persisted wrapper pin replay;
* the ordered admitted row identity, envelope, endpoint-alternative, signature
  and ingress joins;
* the sidecar (saved V2 source-tap surface) replays: one entry per present row,
  payload digests, source/target shapes, ``-1`` border sentinel and index
  bounds;
* all six 1,928-row disposition/refusal cross-tabs, the 957 UNKNOWN (refused)
  rows and the missing-witness / cap intersections.

What it deliberately does NOT do:

* it emits **no** ownership, correctness, recall or accuracy percentage;
* it reads no evaluator payload and never replays a signature, a matcher or a
  solver. The frozen record persists only ``witness_sha256`` for each bound
  source-tap binding, never the ``SignatureSamplingWitnessV1`` tap tables
  (``cell_rectangles`` / ``cell_source_indices`` / rectification tap tables)
  that ``project_ownership_into_source_space`` consumes. That missing payload is
  reported as the exact typed seam and stops the ownership authorization rather
  than re-running screenshots.

Caps are enforced before allocation: the record is opened once (no-follow /
reparse-checked, ``lstat``/``fstat`` identity-bound) and read in bounded chunks
capped at ``max_input_bytes + 1``, JSON depth and token counts are scanned on the
raw bytes before ``json.loads``, and row/witness/page/payload-element/payload-
byte/depth/work/time caps plus the exact nested ``values`` row/element counts,
exact JSON-integer leaves, the contract numeric/sentinel range and the exact
``<i8`` contract dtype are checked on the plain parsed document before any typed
V2 record or NumPy array is constructed.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from mc_photogrammetry.pipeline import candidate_support_provenance as v1
from mc_photogrammetry.pipeline import candidate_support_provenance_v2 as v2
from mc_photogrammetry.pipeline import signature_sampling_persistence as persistence
from mc_photogrammetry.pipeline.frame_ingress import FrameIngressProvenanceV1
from mc_photogrammetry.reconstruction import plane_graph

PREFLIGHT_SCHEMA = "mc-photogrammetry-same-axis-ownership-preflight-v1"

STATUS_COMPLETE = "complete"
STATUS_REFUSED = "refused"

# The accepted V1 record is read as legacy with the sampling payload typed
# unavailable; V2 carries the independently versioned sampling-persistence
# section that this preflight validates for availability.
RECORD_SCHEMA_V1 = "mc-photogrammetry-route-b-diagnostic-record-v1"
RECORD_SCHEMA_V2 = "mc-photogrammetry-route-b-diagnostic-record-v2"
ACCEPTED_RECORD_SCHEMAS = (RECORD_SCHEMA_V1, RECORD_SCHEMA_V2)

# --- Hard input / work caps (checked before allocation) -----------------------
MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_JSON_DEPTH = 16
MAX_JSON_TOKENS = 6_000_000
MAX_UNIVERSE_ROWS = 1_928
MAX_SIGNATURE_WITNESSES = 85
MAX_INDEX_PAGES = 8
MAX_DIGEST_PAGES = 12
MAX_SIDECAR_PAGES = 67
MAX_PAYLOAD_ELEMENTS = 6_000_000
MAX_INGRESS_RECORDS = 8
MAX_WALL_SECONDS = 60.0

# The exact serialized contract dtype for every persisted payload array. The V1
# and V2 contracts require ``np.int64`` (``candidate_support_provenance`` and
# ``candidate_support_provenance_v2`` both compare ``dtype != np.int64``), whose
# ``ndarray.dtype.str`` is ``"<i8"`` on this little-endian platform; the frozen
# record persists exactly ``"<i8"`` for all 2,913 payload arrays. Only that exact
# string is the contract: NumPy aliases such as ``"int64"``/``"i8"``/``"=i8"``/
# ``">i8"`` are deliberately NOT broadened into it, and an unvalidated string is
# never handed to ``np.dtype`` (a tampered ``U1048576`` on a valid ``[147, 4]``
# array would otherwise request ~2.47 GB).
PAYLOAD_CONTRACT_DTYPE = "<i8"
PAYLOAD_CONTRACT_ITEM_BYTES = 8
# Aggregate and per-array payload-byte caps. 6,000,000 contract elements at 8
# bytes each is 48,000,000 bytes; the frozen record measures 4,576,118 elements /
# 36,608,944 contract bytes with a largest single array of 59,232 bytes.
MAX_PAYLOAD_BYTES = MAX_PAYLOAD_ELEMENTS * PAYLOAD_CONTRACT_ITEM_BYTES

# The single-run accepted Route B sidecar cap the record was captured under.
SIDECAR_ENTRY_MAX_RAW_BYTES = 131_072

# Default bounded read chunk: an opened record is never pulled into memory in
# one unbounded call, and never grows past ``max_input_bytes + 1`` bytes.
_READ_CHUNK_BYTES = 1024 * 1024

# --- Bounded typed refusal reasons -------------------------------------------
REASON_INPUT_OVER_CAP = "input_over_cap"
REASON_INPUT_UNREADABLE = "input_unreadable"
REASON_INPUT_SIZE_DRIFT = "input_size_drift"
REASON_DEPTH_OVER_CAP = "depth_over_cap"
REASON_WORK_OVER_CAP = "work_over_cap"
REASON_TIME_OVER_CAP = "time_over_cap"
REASON_RECORD_MALFORMED = "record_malformed"
REASON_RECORD_PIN_MISMATCH = "record_pin_mismatch"
REASON_ARTIFACT_TAMPER = "artifact_tamper"
REASON_ROW_CAP_EXCEEDED = "row_cap_exceeded"
REASON_PAGE_CAP_EXCEEDED = "page_cap_exceeded"
REASON_WITNESS_CAP_EXCEEDED = "witness_cap_exceeded"
REASON_INGRESS_CAP_EXCEEDED = "ingress_cap_exceeded"
REASON_PAYLOAD_OVER_CAP = "payload_over_cap"
REASON_PAYLOAD_DTYPE_INVALID = "payload_dtype_invalid"
REASON_PAYLOAD_VALUE_INVALID = "payload_value_invalid"
REASON_DUPLICATE_ROW_IDENTITY = "duplicate_row_identity"
REASON_DUPLICATE_SIGNATURE = "duplicate_signature"
REASON_DUPLICATE_WITNESS = "duplicate_witness"
REASON_DUPLICATE_INGRESS = "duplicate_ingress"
REASON_ORIENTATION_MISMATCH = "orientation_mismatch"
REASON_INDEX_REPLAY_MISMATCH = "index_replay_mismatch"
REASON_DIGEST_REPLAY_MISMATCH = "digest_replay_mismatch"
REASON_SIDECAR_REPLAY_MISMATCH = "sidecar_replay_mismatch"
REASON_CAPTURE_VALIDATION_FAILED = "capture_validation_failed"
REASON_IDENTITY_JOIN_MISMATCH = "identity_join_mismatch"
REASON_ENVELOPE_JOIN_MISMATCH = "envelope_join_mismatch"
REASON_ENDPOINT_ALTERNATIVE_MISMATCH = "endpoint_alternative_mismatch"
REASON_SIGNATURE_JOIN_MISMATCH = "signature_join_mismatch"
REASON_INGRESS_JOIN_MISMATCH = "ingress_join_mismatch"
REASON_SOURCE_TAP_RECONSTRUCTION_UNAVAILABLE = "source_tap_reconstruction_unavailable"
REASON_SENTINEL_INVALID = "sentinel_invalid"
REASON_TAP_INDEX_INVALID = "tap_index_invalid"
REASON_CROSS_TAB_MISMATCH = "cross_tab_mismatch"
REASON_SAMPLING_PERSISTENCE_INVALID = "sampling_persistence_invalid"
REASON_EVALUATOR_ACCESS_BEFORE_INFERENCE_DIGEST = (
    "evaluator_access_before_inference_digest")

REFUSAL_REASONS = (
    REASON_INPUT_OVER_CAP, REASON_INPUT_UNREADABLE, REASON_INPUT_SIZE_DRIFT,
    REASON_DEPTH_OVER_CAP, REASON_WORK_OVER_CAP, REASON_TIME_OVER_CAP,
    REASON_RECORD_MALFORMED, REASON_RECORD_PIN_MISMATCH, REASON_ARTIFACT_TAMPER,
    REASON_ROW_CAP_EXCEEDED, REASON_PAGE_CAP_EXCEEDED,
    REASON_WITNESS_CAP_EXCEEDED, REASON_INGRESS_CAP_EXCEEDED,
    REASON_PAYLOAD_OVER_CAP, REASON_PAYLOAD_DTYPE_INVALID,
    REASON_PAYLOAD_VALUE_INVALID,
    REASON_DUPLICATE_ROW_IDENTITY, REASON_DUPLICATE_SIGNATURE,
    REASON_DUPLICATE_WITNESS, REASON_DUPLICATE_INGRESS,
    REASON_ORIENTATION_MISMATCH, REASON_INDEX_REPLAY_MISMATCH,
    REASON_DIGEST_REPLAY_MISMATCH, REASON_SIDECAR_REPLAY_MISMATCH,
    REASON_CAPTURE_VALIDATION_FAILED, REASON_IDENTITY_JOIN_MISMATCH,
    REASON_ENVELOPE_JOIN_MISMATCH, REASON_ENDPOINT_ALTERNATIVE_MISMATCH,
    REASON_SIGNATURE_JOIN_MISMATCH, REASON_INGRESS_JOIN_MISMATCH,
    REASON_SOURCE_TAP_RECONSTRUCTION_UNAVAILABLE, REASON_SENTINEL_INVALID,
    REASON_TAP_INDEX_INVALID, REASON_CROSS_TAB_MISMATCH,
    REASON_SAMPLING_PERSISTENCE_INVALID,
    REASON_EVALUATOR_ACCESS_BEFORE_INFERENCE_DIGEST,
)

# --- Required per-frame alignment pins (inventory only; never fabricated) -----
PIN_CAMERA = "camera"
PIN_INTRINSICS = "intrinsics"
PIN_POSE = "pose"
PIN_SOURCE_TRANSFORM = "source_transform"
PIN_OCCLUSION = "occlusion_convention"
PIN_EVALUATOR_DOMAIN = "evaluator_domain"
REQUIRED_ALIGNMENT_PINS = (
    PIN_CAMERA, PIN_INTRINSICS, PIN_POSE, PIN_SOURCE_TRANSFORM,
    PIN_OCCLUSION, PIN_EVALUATOR_DOMAIN)

PIN_STATE_AVAILABLE = "available"
PIN_STATE_UNAVAILABLE = "unavailable"

# The exact, bounded seam name for the missing Phase-B per-cell tap payload.
SAMPLING_WITNESS_TAP_SEAM = (
    "signature_witnesses[].witness_sha256 has no persisted "
    "SignatureSamplingWitnessV1 payload (cell_rectangles/cell_source_indices/"
    "rectification tap tables)")
# The V2 section carries the payload; when it is validated and every bound
# witness decodes intact the seam closes (availability), while ownership stays
# withheld for the missing camera/intrinsics/pose/occlusion/evaluator pins.
SAMPLING_WITNESS_SECTION_INVALID = (
    "sampling_persistence section is present but not fully decodable")
OWNERSHIP_WITHHELD = "withheld"
OWNERSHIP_WITHHELD_REASON = "alignment_prerequisites_unavailable"


class SameAxisOwnershipPreflightError(ValueError):
    """Typed bounded refusal raised internally and surfaced as a record."""

    def __init__(self, reason: str, field: str = "", detail: str = "") -> None:
        if reason not in REFUSAL_REASONS:
            reason = REASON_RECORD_MALFORMED
        self.reason = reason
        self.field = field
        self.detail = detail
        super().__init__(reason if not detail else f"{reason}: {detail}")


class EvaluatorAccessRefused(SameAxisOwnershipPreflightError):
    """Raised when evaluator-domain access precedes the inference digest seal."""

    def __init__(self, field: str = "evaluator") -> None:
        super().__init__(REASON_EVALUATOR_ACCESS_BEFORE_INFERENCE_DIGEST, field)


# --- Canonical helpers --------------------------------------------------------

def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _hex64(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _is_positive_int(value: object) -> bool:
    return type(value) is int and value > 0


# --- Caps and limits ----------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PreflightLimitsV1:
    max_input_bytes: int = MAX_INPUT_BYTES
    max_json_depth: int = MAX_JSON_DEPTH
    max_json_tokens: int = MAX_JSON_TOKENS
    max_universe_rows: int = MAX_UNIVERSE_ROWS
    max_signature_witnesses: int = MAX_SIGNATURE_WITNESSES
    max_index_pages: int = MAX_INDEX_PAGES
    max_digest_pages: int = MAX_DIGEST_PAGES
    max_sidecar_pages: int = MAX_SIDECAR_PAGES
    max_payload_elements: int = MAX_PAYLOAD_ELEMENTS
    max_payload_bytes: int = MAX_PAYLOAD_BYTES
    max_payload_array_bytes: int = MAX_PAYLOAD_BYTES
    max_ingress_records: int = MAX_INGRESS_RECORDS
    max_wall_seconds: float = MAX_WALL_SECONDS


class _Deadline:
    __slots__ = ("_clock", "_deadline")

    def __init__(self, seconds: float, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._deadline = clock() + float(seconds)

    def check(self) -> None:
        if self._clock() > self._deadline:
            raise SameAxisOwnershipPreflightError(
                REASON_TIME_OVER_CAP, "wall_clock")


# --- Bounded input read (byte cap + depth/work scan before json.loads) --------

@dataclass(frozen=True, slots=True)
class BoundedRecordBytesV1:
    data: bytes
    sha256: str
    byte_length: int
    max_json_depth: int
    json_tokens: int


def scan_json_bounds(data: bytes) -> tuple[int, int]:
    """Return ``(max_nesting_depth, token_count)`` without allocating objects.

    Strings and escapes are skipped so brackets inside a string value never
    count. The scan is a single linear pass over the raw bytes; it is the
    depth/work gate that runs *before* ``json.loads``.
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
                raise SameAxisOwnershipPreflightError(
                    REASON_RECORD_MALFORMED, "json_structure")
        elif byte in (0x2C, 0x3A):      # , :
            tokens += 1
    if in_string or depth != 0:
        raise SameAxisOwnershipPreflightError(
            REASON_RECORD_MALFORMED, "json_structure")
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
        chunk = stream.read(min(_READ_CHUNK_BYTES, max_bytes + 1 - len(chunks)))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)


def _read_bounded_file_snapshot(path: Path, max_bytes: int,
                                deadline: _Deadline) -> bytes:
    """Read one stable opened-file snapshot capped at ``max_bytes``.

    This mirrors the reviewed ``runtime_render_provenance`` bounded snapshot
    contract (no-follow/reparse rejection, ``lstat``/``fstat`` identity, chunked
    reads capped at ``max_bytes + 1``, post-read identity re-check), but keeps
    this module's typed refusal reasons and injected wall-clock deadline, which
    that helper's fixed ``ProvenanceError`` taxonomy cannot express. The file is
    read through one opened descriptor, so a concurrent grow/replace can never
    make the reader allocate more than ``max_bytes + 1`` bytes.
    """
    path = Path(path)
    if _linklike_path(path):
        raise SameAxisOwnershipPreflightError(
            REASON_INPUT_UNREADABLE, "record.json", "link_or_reparse_point")
    try:
        before_path = path.lstat()
    except OSError as error:
        raise SameAxisOwnershipPreflightError(
            REASON_INPUT_UNREADABLE, "record.json", type(error).__name__) from error
    if not stat.S_ISREG(before_path.st_mode):
        raise SameAxisOwnershipPreflightError(
            REASON_INPUT_UNREADABLE, "record.json", "not_regular_file")
    if before_path.st_size > max_bytes:
        raise SameAxisOwnershipPreflightError(REASON_INPUT_OVER_CAP, "record.json")
    deadline.check()
    flags = (os.O_RDONLY | getattr(os, "O_BINARY", 0)
             | getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SameAxisOwnershipPreflightError(
            REASON_INPUT_UNREADABLE, "record.json", type(error).__name__) from error
    try:
        try:
            before_fd = os.fstat(descriptor)
            if not _same_file_identity(before_path, before_fd):
                raise SameAxisOwnershipPreflightError(
                    REASON_INPUT_SIZE_DRIFT, "record.json")
            chunks = bytearray()
            while len(chunks) <= max_bytes:
                deadline.check()
                chunk = os.read(
                    descriptor, min(_READ_CHUNK_BYTES, max_bytes + 1 - len(chunks)))
                if not chunk:
                    break
                chunks.extend(chunk)
            raw = bytes(chunks)
            after_fd = os.fstat(descriptor)
        except OSError as error:
            raise SameAxisOwnershipPreflightError(
                REASON_INPUT_UNREADABLE, "record.json",
                type(error).__name__) from error
    finally:
        os.close(descriptor)
    try:
        after_path = path.lstat()
    except OSError as error:
        raise SameAxisOwnershipPreflightError(
            REASON_INPUT_SIZE_DRIFT, "record.json", type(error).__name__) from error
    if len(raw) > max_bytes:
        raise SameAxisOwnershipPreflightError(REASON_INPUT_OVER_CAP, "record.json")
    if (len(raw) != before_path.st_size
            or not _same_file_identity(before_fd, after_fd)
            or not _same_file_identity(before_path, after_path)):
        raise SameAxisOwnershipPreflightError(
            REASON_INPUT_SIZE_DRIFT, "record.json")
    return raw


def read_bounded_record_bytes(
    path: "str | os.PathLike[str]",
    limits: PreflightLimitsV1 = PreflightLimitsV1(),
    *,
    reader: Callable[[object], bytes] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> BoundedRecordBytesV1:
    """Read one record under the declared caps, refusing before allocation.

    By default the record is read through a single opened, non-link/reparse
    regular-file descriptor whose ``lstat``/``fstat`` identity and size must
    agree before and after a chunked read capped at ``max_input_bytes + 1``; a
    file that grows or is replaced mid-read is refused instead of allocated.
    This is a stable-read identity gate on this platform, not an atomic snapshot:
    a writer that overwrites in place with the same size and restores the recorded
    mtime before the final ``lstat`` cannot be distinguished by these checks (the
    same residual limitation is documented on the persistence module's reader).
    An injected ``reader`` keeps the same metadata pre-cap and post-read byte
    checks. Depth and token scans run on the raw bytes before ``json.loads``.
    """
    deadline = _Deadline(limits.max_wall_seconds, clock)
    path = Path(path)
    if reader is not None:
        try:
            size = path.lstat().st_size
        except OSError as error:
            raise SameAxisOwnershipPreflightError(
                REASON_INPUT_UNREADABLE, "record.json",
                type(error).__name__) from error
        if size > limits.max_input_bytes:
            raise SameAxisOwnershipPreflightError(
                REASON_INPUT_OVER_CAP, "record.json")
        deadline.check()
        try:
            data = reader(path)
        except OSError as error:
            raise SameAxisOwnershipPreflightError(
                REASON_INPUT_UNREADABLE, "record.json",
                type(error).__name__) from error
        if type(data) is not bytes:
            raise SameAxisOwnershipPreflightError(
                REASON_INPUT_UNREADABLE, "record.json")
        if len(data) != size:
            raise SameAxisOwnershipPreflightError(
                REASON_INPUT_SIZE_DRIFT, "record.json")
        if len(data) > limits.max_input_bytes:
            raise SameAxisOwnershipPreflightError(
                REASON_INPUT_OVER_CAP, "record.json")
    else:
        data = _read_bounded_file_snapshot(path, limits.max_input_bytes, deadline)
    deadline.check()
    max_depth, tokens = scan_json_bounds(data)
    if max_depth > limits.max_json_depth:
        raise SameAxisOwnershipPreflightError(REASON_DEPTH_OVER_CAP, "record.json")
    if tokens > limits.max_json_tokens:
        raise SameAxisOwnershipPreflightError(REASON_WORK_OVER_CAP, "record.json")
    deadline.check()
    return BoundedRecordBytesV1(
        data=data, sha256=hashlib.sha256(data).hexdigest(), byte_length=len(data),
        max_json_depth=max_depth, json_tokens=tokens)


# --- Frozen Route B expectations ---------------------------------------------

@dataclass(frozen=True, slots=True)
class FrozenRouteBExpectationsV1:
    input_bytes: int = 16_896_513
    input_sha256: str = (
        "006e8b6b16a312cdaa61a17b4336ce27c1abac590435645908ea9cfef2e998ee")
    artifact_sha256: str = (
        "678475d64a95a2814b5cce6864cd55853cc8f8b5e45cdf8d1b86a7023d5d41af")
    record_sha256: str = (
        "6eddf7f2bb60c1bdeac053d86e592dcd3c5c1b4bdd91a056c960c6f8db8ba797")
    whole_universe_identity_sha256: str = (
        "a5ddc8a65c0414e99b6026b5c2d7f4770e942f495ac9ebd8f26e59368d9c0efc")
    input_match_identity_sha256: str = (
        "28ad4ac48021c727c4b6e7bf283e9965514dc3016c3c16919a18367cf0868105")
    failed_assignment_sha256: str = (
        "ab4cb09bf5e8fe00e0bca11ca6cf2732cf24fd086d6f717919369f4191525dcd")
    expected_row_count: int = 1_928
    signature_witness_count: int = 85
    index_page_count: int = 5
    digest_page_count: int = 6
    sidecar_page_count: int = 67
    sidecar_entry_count: int = 971
    present_row_count: int = 971
    unknown_row_count: int = 957
    kept_row_count: int = 38
    dropped_row_count: int = 1_890
    group_count: int = 35
    bound_witness_count: int = 66
    refused_witness_count: int = 19
    referenced_signature_count: int = 78
    referenced_bound_count: int = 59
    referenced_unbound_count: int = 19
    bound_unreferenced_count: int = 7
    missing_witness_row_count: int = 525
    missing_witness_over_cap_count: int = 412
    missing_witness_refusal_count: int = 113
    raw_cap_provable_count: int = 102
    ingress_frame_ids: tuple[str, ...] = ("image-0000", "image-0001", "image-0002")
    cross_tabs: tuple[tuple[str, str, "str | None", int], ...] = (
        ("kept", "present", None, 26),
        ("kept", "refused", "support_payload_over_cap", 11),
        ("kept", "refused", "source_tap_witness_unavailable", 1),
        ("same_axis_mismatch", "present", None, 945),
        ("same_axis_mismatch", "refused", "support_payload_over_cap", 833),
        ("same_axis_mismatch", "refused", "source_tap_witness_unavailable", 112),
    )


FROZEN_ROUTE_B_EXPECTATIONS = FrozenRouteBExpectationsV1()


def frozen_record_path() -> Path:
    """The one accepted frozen record, resolved from this module's repo root."""
    return (Path(__file__).resolve().parents[2]
            / ".tmp-routeb-v2-frozen-r4-20261004" / "record.json")


# --- Typed inventory records --------------------------------------------------

@dataclass(frozen=True, slots=True)
class PinStateV1:
    pin: str
    state: str
    source: str = ""
    detail: str = ""


@dataclass(frozen=True, slots=True)
class FrameAlignmentPinInventoryV1:
    frame_id: str
    pins: tuple[PinStateV1, ...]

    def pin(self, name: str) -> PinStateV1:
        for item in self.pins:
            if item.pin == name:
                return item
        return PinStateV1(name, PIN_STATE_UNAVAILABLE, detail="pin_not_inventoried")


@dataclass(frozen=True, slots=True)
class EndpointJoinFactsV1:
    endpoint_rows: int
    referenced_signature_count: int
    referenced_bound_count: int
    referenced_unbound_count: int
    bound_unreferenced_count: int
    missing_witness_row_count: int
    missing_witness_over_cap_count: int
    missing_witness_refusal_count: int
    raw_cap_provable_count: int


@dataclass(frozen=True, slots=True)
class TapSurfaceFactsV1:
    present_rows: int
    sidecar_entries: int
    payload_elements: int
    border_sentinel_count: int
    logical_pair_count: int


# --- Evaluator access gate ----------------------------------------------------

@dataclass(frozen=True, slots=True)
class InferenceDigestSealV1:
    artifact_sha256: str
    record_sha256: str
    whole_universe_identity_sha256: str
    input_match_identity_sha256: str
    failed_assignment_sha256: str


class InferenceDigestGate:
    """Blocks evaluator-domain access until the frozen inference digest is sealed.

    The preflight seals this gate only after the frozen record bytes, artifact
    digest, V2 root digest and the identity digests have all replayed. Evaluator
    inventory labels are then produced from that frozen inference state; asking
    for evaluator access earlier is a typed refusal rather than a best-effort
    read.
    """

    __slots__ = ("_seal",)

    def __init__(self) -> None:
        self._seal: InferenceDigestSealV1 | None = None

    @property
    def sealed(self) -> bool:
        return self._seal is not None

    @property
    def seal(self) -> InferenceDigestSealV1 | None:
        return self._seal

    def seal_inference_digest(self, *, artifact_sha256: str, record_sha256: str,
                              whole_universe_identity_sha256: str,
                              input_match_identity_sha256: str,
                              failed_assignment_sha256: str) -> InferenceDigestSealV1:
        for value in (artifact_sha256, record_sha256,
                      whole_universe_identity_sha256,
                      input_match_identity_sha256, failed_assignment_sha256):
            if not _hex64(value):
                raise SameAxisOwnershipPreflightError(
                    REASON_RECORD_PIN_MISMATCH, "inference_digest")
        self._seal = InferenceDigestSealV1(
            artifact_sha256=artifact_sha256, record_sha256=record_sha256,
            whole_universe_identity_sha256=whole_universe_identity_sha256,
            input_match_identity_sha256=input_match_identity_sha256,
            failed_assignment_sha256=failed_assignment_sha256)
        return self._seal

    def require_evaluator_access(self, field: str = "evaluator") -> InferenceDigestSealV1:
        if self._seal is None:
            raise EvaluatorAccessRefused(field)
        return self._seal


# --- Result -------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SameAxisOwnershipPreflightV1:
    status: str
    reason: str = ""
    field: str = ""
    detail: str = ""
    input_bytes: int = 0
    input_sha256: str = ""
    max_json_depth: int = 0
    json_tokens: int = 0
    record_schema: str = ""
    record_status: str = ""
    row_count: int = 0
    present_row_count: int = 0
    unknown_row_count: int = 0
    kept_row_count: int = 0
    dropped_row_count: int = 0
    group_count: int = 0
    index_page_count: int = 0
    digest_page_count: int = 0
    sidecar_page_count: int = 0
    sidecar_entry_count: int = 0
    payload_element_count: int = 0
    payload_elements_allocated: int = 0
    signature_witness_count: int = 0
    bound_witness_count: int = 0
    refused_witness_count: int = 0
    cross_tabs: tuple[tuple[str, str, "str | None", int], ...] = ()
    endpoint_join: EndpointJoinFactsV1 | None = None
    ingress_frame_ids: tuple[str, ...] = ()
    frame_pins: tuple[FrameAlignmentPinInventoryV1, ...] = ()
    tap_surface_state: str = PIN_STATE_UNAVAILABLE
    sampling_witness_state: str = PIN_STATE_UNAVAILABLE
    sampling_witness_seam: str = ""
    # Present only for a V2 record that carries a validated sampling-persistence
    # section; a V1 record leaves this ``None`` so its result stays byte-identical.
    sampling_persistence: "dict | None" = None
    ownership_authorization: str = OWNERSHIP_WITHHELD
    ownership_blockers: tuple[str, ...] = ()
    gate_sealed: bool = False
    evaluator_payload_read: bool = False
    schema: str = PREFLIGHT_SCHEMA

    @property
    def refused(self) -> bool:
        return self.status == STATUS_REFUSED

    def to_record(self) -> dict:
        def _pin(item: PinStateV1) -> dict:
            return {"pin": item.pin, "state": item.state, "source": item.source,
                    "detail": item.detail}

        record = {
            "schema": self.schema,
            "status": self.status,
            "reason": self.reason,
            "field": self.field,
            "detail": self.detail,
            "input_bytes": self.input_bytes,
            "input_sha256": self.input_sha256,
            "max_json_depth": self.max_json_depth,
            "json_tokens": self.json_tokens,
            "record_schema": self.record_schema,
            "record_status": self.record_status,
            "row_count": self.row_count,
            "present_row_count": self.present_row_count,
            "unknown_row_count": self.unknown_row_count,
            "kept_row_count": self.kept_row_count,
            "dropped_row_count": self.dropped_row_count,
            "group_count": self.group_count,
            "index_page_count": self.index_page_count,
            "digest_page_count": self.digest_page_count,
            "sidecar_page_count": self.sidecar_page_count,
            "sidecar_entry_count": self.sidecar_entry_count,
            "payload_element_count": self.payload_element_count,
            "payload_elements_allocated": self.payload_elements_allocated,
            "signature_witness_count": self.signature_witness_count,
            "bound_witness_count": self.bound_witness_count,
            "refused_witness_count": self.refused_witness_count,
            "cross_tabs": [
                {"disposition": disposition, "support_state": support_state,
                 "support_refusal_reason": refusal_reason, "rows": rows}
                for disposition, support_state, refusal_reason, rows
                in self.cross_tabs],
            "endpoint_join": (None if self.endpoint_join is None else {
                "endpoint_rows": self.endpoint_join.endpoint_rows,
                "referenced_signature_count":
                    self.endpoint_join.referenced_signature_count,
                "referenced_bound_count": self.endpoint_join.referenced_bound_count,
                "referenced_unbound_count":
                    self.endpoint_join.referenced_unbound_count,
                "bound_unreferenced_count":
                    self.endpoint_join.bound_unreferenced_count,
                "missing_witness_row_count":
                    self.endpoint_join.missing_witness_row_count,
                "missing_witness_over_cap_count":
                    self.endpoint_join.missing_witness_over_cap_count,
                "missing_witness_refusal_count":
                    self.endpoint_join.missing_witness_refusal_count,
                "raw_cap_provable_count":
                    self.endpoint_join.raw_cap_provable_count}),
            "ingress_frame_ids": list(self.ingress_frame_ids),
            "frame_pins": [
                {"frame_id": inventory.frame_id,
                 "pins": [_pin(item) for item in inventory.pins]}
                for inventory in self.frame_pins],
            "tap_surface_state": self.tap_surface_state,
            "sampling_witness_state": self.sampling_witness_state,
            "sampling_witness_seam": self.sampling_witness_seam,
            "ownership_authorization": self.ownership_authorization,
            "ownership_blockers": list(self.ownership_blockers),
            "gate_sealed": self.gate_sealed,
            "evaluator_payload_read": self.evaluator_payload_read,
        }
        if self.sampling_persistence is not None:
            record["sampling_persistence"] = self.sampling_persistence
        return record


def _refused_result(reason: str, field: str, detail: str,
                    *, bounded: BoundedRecordBytesV1 | None = None,
                    record_schema: str = "", record_status: str = "") -> SameAxisOwnershipPreflightV1:
    return SameAxisOwnershipPreflightV1(
        status=STATUS_REFUSED, reason=reason, field=field, detail=detail,
        input_bytes=(0 if bounded is None else bounded.byte_length),
        input_sha256=("" if bounded is None else bounded.sha256),
        max_json_depth=(0 if bounded is None else bounded.max_json_depth),
        json_tokens=(0 if bounded is None else bounded.json_tokens),
        record_schema=record_schema, record_status=record_status)


# --- Structural prescan (caps before typed/numpy allocation) ------------------

def _as_dict(value: object) -> dict:
    if type(value) is not dict:
        raise SameAxisOwnershipPreflightError(
            REASON_RECORD_MALFORMED, "record")
    return value


def _as_list(value: object, field: str) -> list:
    if type(value) is not list:
        raise SameAxisOwnershipPreflightError(
            REASON_RECORD_MALFORMED, field)
    return value


def _row_identity_tuple(row: Mapping[str, object]) -> tuple:
    try:
        return (row["left"], row["left_plane"], row["left_axis"], row["right"],
                row["right_plane"], row["right_axis"], row["quadrant_of_right"],
                float(row["left_b"]), float(row["right_b"]))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise SameAxisOwnershipPreflightError(
            REASON_RECORD_MALFORMED, "index_row", type(error).__name__) from error


def _shape_elements(shape: object, field: str) -> int:
    if (type(shape) is not list or not shape
            or any(type(item) is not int or item < 0 for item in shape)):
        raise SameAxisOwnershipPreflightError(
            REASON_RECORD_MALFORMED, field)
    product = 1
    for item in shape:
        product *= item
    return product


def _validate_nested_values(values: object, shape: list, field: str,
                            element_cap: int, *, minimum_value: int,
                            maximum_value: int,
                            sentinel_allowed: bool) -> int:
    """Validate nested ``values`` against declared ``shape``; return leaf count.

    Every level must be a JSON list whose length equals the declared dimension
    and every leaf must be an exact JSON integer inside the contract range.
    ``bool`` is rejected (``type(item) is int`` is false for it), so a float or
    boolean leaf can never be silently truncated or coerced by the ``<i8``
    conversion. Range/sentinel refusals use the same typed reasons the post-
    allocation NumPy checks used. Counting stops as soon as ``element_cap`` is
    exceeded, so an over-cap, mismatched or malformed nesting is refused here,
    before NumPy is ever asked to allocate or convert it.
    """
    if len(shape) > MAX_JSON_DEPTH:
        raise SameAxisOwnershipPreflightError(
            REASON_RECORD_MALFORMED, f"{field}.shape")
    last = len(shape) - 1
    counted = 0
    stack: list[tuple[object, int]] = [(values, 0)]
    while stack:
        node, depth = stack.pop()
        if type(node) is not list or len(node) != shape[depth]:
            raise SameAxisOwnershipPreflightError(
                REASON_SIDECAR_REPLAY_MISMATCH, field)
        if depth == last:
            for item in node:
                if type(item) is list:
                    raise SameAxisOwnershipPreflightError(
                        REASON_SIDECAR_REPLAY_MISMATCH, field)
                if type(item) is not int:
                    raise SameAxisOwnershipPreflightError(
                        REASON_PAYLOAD_VALUE_INVALID, field)
                if item < minimum_value:
                    raise SameAxisOwnershipPreflightError(
                        REASON_SENTINEL_INVALID if sentinel_allowed
                        else REASON_TAP_INDEX_INVALID, field)
                if item > maximum_value:
                    raise SameAxisOwnershipPreflightError(
                        REASON_TAP_INDEX_INVALID, field)
                counted += 1
                if counted > element_cap:
                    raise SameAxisOwnershipPreflightError(
                        REASON_PAYLOAD_OVER_CAP, field)
        else:
            child_depth = depth + 1
            for item in node:
                stack.append((item, child_depth))
    return counted


def _prescan_structure(document: dict, limits: PreflightLimitsV1) -> dict:
    """Cap and duplicate checks on the plain parsed document.

    This runs before any typed V2 record or NumPy array is constructed. It keeps
    only counts and the admitted-row ordinal/identity keys it must compare, so a
    tampered record cannot force an over-cap allocation.
    """
    if document.get("schema") not in ACCEPTED_RECORD_SCHEMAS:
        raise SameAxisOwnershipPreflightError(
            REASON_RECORD_MALFORMED, "record.schema")
    if document.get("status") != "complete":
        raise SameAxisOwnershipPreflightError(
            REASON_RECORD_MALFORMED, "record.status")
    capture = _as_dict(document.get("capture"))
    if capture.get("schema") != v2.COMPLETE_SUPPORT_CAPTURE_SCHEMA:
        raise SameAxisOwnershipPreflightError(
            REASON_RECORD_MALFORMED, "capture.schema")
    root = _as_dict(capture.get("root"))

    index_pages = _as_list(capture.get("index_pages"), "capture.index_pages")
    digest_pages = _as_list(capture.get("digest_pages"), "capture.digest_pages")
    sidecar_pages = _as_list(capture.get("sidecar_pages"), "capture.sidecar_pages")
    if len(index_pages) > limits.max_index_pages:
        raise SameAxisOwnershipPreflightError(
            REASON_PAGE_CAP_EXCEEDED, "index_pages")
    if len(digest_pages) > limits.max_digest_pages:
        raise SameAxisOwnershipPreflightError(
            REASON_PAGE_CAP_EXCEEDED, "digest_pages")
    if len(sidecar_pages) > limits.max_sidecar_pages:
        raise SameAxisOwnershipPreflightError(
            REASON_PAGE_CAP_EXCEEDED, "sidecar_pages")

    row_count = 0
    seen_ordinals: set[int] = set()
    seen_identities: set[tuple] = set()
    present_ordinals: list[int] = []
    group_keys: set[tuple] = set()
    for page_index, page in enumerate(index_pages):
        page = _as_dict(page)
        rows = _as_list(page.get("rows"), f"index_pages.{page_index}.rows")
        for row in rows:
            row = _as_dict(row)
            ordinal = row.get("admitted_row_index")
            if type(ordinal) is not int or ordinal < 0:
                raise SameAxisOwnershipPreflightError(
                    REASON_RECORD_MALFORMED, "admitted_row_index")
            if ordinal in seen_ordinals:
                raise SameAxisOwnershipPreflightError(
                    REASON_DUPLICATE_ROW_IDENTITY, "admitted_row_index")
            seen_ordinals.add(ordinal)
            identity = _row_identity_tuple(row)
            if identity in seen_identities:
                raise SameAxisOwnershipPreflightError(
                    REASON_DUPLICATE_ROW_IDENTITY, "row_identity")
            seen_identities.add(identity)
            if row.get("orientation") != v1.CANDIDATE_SUPPORT_ORIENTATION:
                raise SameAxisOwnershipPreflightError(
                    REASON_ORIENTATION_MISMATCH, f"row.{ordinal}.orientation")
            if row.get("support_state") == v2.SUPPORT_STATE_PRESENT:
                present_ordinals.append(ordinal)
            group_keys.add((row.get("left"), row.get("right"),
                            row.get("left_axis"), row.get("quadrant_of_right")))
            row_count += 1
            if row_count > limits.max_universe_rows:
                raise SameAxisOwnershipPreflightError(
                    REASON_ROW_CAP_EXCEEDED, "row_count")
    if row_count == 0:
        raise SameAxisOwnershipPreflightError(
            REASON_RECORD_MALFORMED, "row_count")
    if sorted(seen_ordinals) != list(range(row_count)):
        raise SameAxisOwnershipPreflightError(
            REASON_DUPLICATE_ROW_IDENTITY, "admitted_row_index")

    section_lengths: dict[str, int] = {}
    for page_index, page in enumerate(digest_pages):
        page = _as_dict(page)
        section = page.get("section")
        if section not in v2.DIGEST_SECTIONS:
            raise SameAxisOwnershipPreflightError(
                REASON_DIGEST_REPLAY_MISMATCH, f"digest_pages.{page_index}.section")
        digests = _as_list(page.get("digests"),
                           f"digest_pages.{page_index}.digests")
        section_lengths[section] = section_lengths.get(section, 0) + len(digests)
    for section in v2.DIGEST_SECTIONS:
        if section_lengths.get(section) != row_count:
            raise SameAxisOwnershipPreflightError(
                REASON_DIGEST_REPLAY_MISMATCH, f"section.{section}")

    sidecar_entries = 0
    payload_elements = 0
    payload_bytes = 0
    entry_ordinals: list[int] = []
    seen_entry_ordinals: set[int] = set()
    for page_index, page in enumerate(sidecar_pages):
        page = _as_dict(page)
        entries = _as_list(page.get("entries"),
                           f"sidecar_pages.{page_index}.entries")
        for entry in entries:
            entry = _as_dict(entry)
            ordinal = entry.get("admitted_row_index")
            if type(ordinal) is not int or ordinal < 0:
                raise SameAxisOwnershipPreflightError(
                    REASON_RECORD_MALFORMED, "sidecar.admitted_row_index")
            if ordinal in seen_entry_ordinals:
                raise SameAxisOwnershipPreflightError(
                    REASON_SIDECAR_REPLAY_MISMATCH, "sidecar.duplicate_entry")
            seen_entry_ordinals.add(ordinal)
            entry_ordinals.append(ordinal)
            payload = _as_dict(entry.get("payload"))
            for name in ("logical_pairs", "warp_source_rows", "warp_source_columns"):
                array = _as_dict(payload.get(name))
                field = f"payload.{name}"
                # Exact contract dtype first: an unvalidated dtype string must
                # never reach ``np.dtype``/``np.asarray``.
                if array.get("dtype") != PAYLOAD_CONTRACT_DTYPE:
                    raise SameAxisOwnershipPreflightError(
                        REASON_PAYLOAD_DTYPE_INVALID, f"{field}.dtype")
                shape = array.get("shape")
                elements = _shape_elements(shape, f"{field}.shape")
                remaining = limits.max_payload_elements - payload_elements
                if elements > remaining:
                    raise SameAxisOwnershipPreflightError(
                        REASON_PAYLOAD_OVER_CAP, "payload_elements")
                array_bytes = elements * PAYLOAD_CONTRACT_ITEM_BYTES
                if array_bytes > limits.max_payload_array_bytes:
                    raise SameAxisOwnershipPreflightError(
                        REASON_PAYLOAD_OVER_CAP, f"{field}.bytes")
                if array_bytes > limits.max_payload_bytes - payload_bytes:
                    raise SameAxisOwnershipPreflightError(
                        REASON_PAYLOAD_OVER_CAP, "payload_bytes")
                # ``-1`` is the documented border sentinel for the two warp maps
                # only; ``logical_pairs`` is a real selected-pair index list and
                # never accepts it.
                sentinel_allowed = name != "logical_pairs"
                counted = _validate_nested_values(
                    array.get("values"), shape, f"{field}.values",
                    element_cap=remaining,
                    minimum_value=(-1 if sentinel_allowed else 0),
                    maximum_value=v2.PAYLOAD_MAX_ABS_ELEMENT,
                    sentinel_allowed=sentinel_allowed)
                if counted != elements:
                    raise SameAxisOwnershipPreflightError(
                        REASON_SIDECAR_REPLAY_MISMATCH, f"{field}.values")
                payload_elements += counted
                payload_bytes += array_bytes
            sidecar_entries += 1
            if sidecar_entries > limits.max_universe_rows:
                raise SameAxisOwnershipPreflightError(
                    REASON_ROW_CAP_EXCEEDED, "sidecar_entries")
    if payload_elements > limits.max_payload_elements:
        raise SameAxisOwnershipPreflightError(
            REASON_PAYLOAD_OVER_CAP, "payload_elements")
    if payload_bytes > limits.max_payload_bytes:
        raise SameAxisOwnershipPreflightError(
            REASON_PAYLOAD_OVER_CAP, "payload_bytes")
    if entry_ordinals != sorted(entry_ordinals):
        raise SameAxisOwnershipPreflightError(
            REASON_SIDECAR_REPLAY_MISMATCH, "sidecar.entry_order")
    if entry_ordinals != present_ordinals:
        # The missing/changed entry is the exact tap-reconstruction seam.
        missing = sorted(set(present_ordinals) - set(entry_ordinals))
        if missing:
            raise SameAxisOwnershipPreflightError(
                REASON_SOURCE_TAP_RECONSTRUCTION_UNAVAILABLE,
                f"sidecar.entry.{missing[0]}")
        raise SameAxisOwnershipPreflightError(
            REASON_SIDECAR_REPLAY_MISMATCH, "sidecar.present_rows")

    witnesses = _as_list(document.get("signature_witnesses"),
                         "signature_witnesses")
    if len(witnesses) > limits.max_signature_witnesses:
        raise SameAxisOwnershipPreflightError(
            REASON_WITNESS_CAP_EXCEEDED, "signature_witnesses")
    ingress = _as_list(document.get("ingress"), "ingress")
    if len(ingress) > limits.max_ingress_records:
        raise SameAxisOwnershipPreflightError(
            REASON_INGRESS_CAP_EXCEEDED, "ingress")

    # The V2 sampling-persistence section is caps-validated on the plain document
    # (declared dtype/shape/bytes, exact canonical base64 length, pointers,
    # weights, rectangles, section digest) before any base64 decode or NumPy
    # allocation. A V1 record is legacy and must not carry the section.
    section_accounting = 0
    if document.get("schema") == RECORD_SCHEMA_V2:
        section = document.get("sampling_persistence")
        if section is None:
            raise SameAxisOwnershipPreflightError(
                REASON_SAMPLING_PERSISTENCE_INVALID, "sampling_persistence")
        try:
            accounting = persistence.validate_section_plain(section)
        except persistence.SamplingPersistenceError as error:
            raise SameAxisOwnershipPreflightError(
                REASON_SAMPLING_PERSISTENCE_INVALID, error.field,
                error.reason) from error
        if accounting.entry_count > limits.max_signature_witnesses:
            raise SameAxisOwnershipPreflightError(
                REASON_WITNESS_CAP_EXCEEDED, "sampling_persistence.entries")
        section_accounting = accounting.entry_count
    elif "sampling_persistence" in document:
        raise SameAxisOwnershipPreflightError(
            REASON_SAMPLING_PERSISTENCE_INVALID, "sampling_persistence")

    return {
        "row_count": row_count,
        "present_ordinals": present_ordinals,
        "group_count": len(group_keys),
        "index_page_count": len(index_pages),
        "digest_page_count": len(digest_pages),
        "sidecar_page_count": len(sidecar_pages),
        "sidecar_entry_count": sidecar_entries,
        "payload_elements": payload_elements,
        "payload_bytes": payload_bytes,
        "sampling_entry_count": section_accounting,
    }


# --- Typed V2 capture rebuild (bounded; after prescan) ------------------------

def _init_fields(cls: type) -> frozenset:
    return frozenset(field.name for field in dataclasses.fields(cls) if field.init)


_SCALAR_SEQUENCES = {"decoded_shape": tuple, "source_shape": tuple,
                     "target_shape": tuple}


def _optional_tuple(value: object):
    return None if value is None else tuple(value)


def _init_kwargs(cls: type, record: object, *, rebuild: Mapping[str, object] = ()) -> dict:
    record = _as_dict(record)
    allowed = _init_fields(cls)
    kwargs: dict = {}
    for name, value in record.items():
        if name not in allowed:
            continue
        if name in rebuild:
            kwargs[name] = rebuild[name]
        elif name in _SCALAR_SEQUENCES:
            kwargs[name] = _SCALAR_SEQUENCES[name](value)
        else:
            kwargs[name] = value
    return kwargs


def _array_from_record(record: object, limits: PreflightLimitsV1,
                       *, allow_border_sentinel: bool) -> "object":
    import numpy as np

    record = _as_dict(record)
    values = record.get("values")
    dtype = record.get("dtype")
    shape = record.get("shape")
    if (type(values) is not list or type(dtype) is not str
            or type(shape) is not list
            or any(type(item) is not int or item < 0 for item in shape)):
        raise SameAxisOwnershipPreflightError(
            REASON_SIDECAR_REPLAY_MISMATCH, "payload_array")
    # Exact serialized contract dtype, checked again immediately before NumPy so
    # an unvalidated string can never reach ``np.dtype``/``np.asarray``.
    if dtype != PAYLOAD_CONTRACT_DTYPE:
        raise SameAxisOwnershipPreflightError(
            REASON_PAYLOAD_DTYPE_INVALID, "payload_array.dtype")
    elements = _shape_elements(shape, "payload_array.shape")
    if elements > limits.max_payload_elements:
        raise SameAxisOwnershipPreflightError(
            REASON_PAYLOAD_OVER_CAP, "payload_array")
    array_bytes = elements * PAYLOAD_CONTRACT_ITEM_BYTES
    if (array_bytes > limits.max_payload_array_bytes
            or array_bytes > limits.max_payload_bytes):
        raise SameAxisOwnershipPreflightError(
            REASON_PAYLOAD_OVER_CAP, "payload_array.bytes")
    # Exact nested row/element counts, exact JSON-integer leaves and the
    # contract numeric/sentinel range are all verified before NumPy is called, so
    # a small declared shape can never smuggle a much larger ``values`` list past
    # the cap into an allocation and a float/bool leaf can never be truncated.
    counted = _validate_nested_values(
        values, shape, "payload_array.values",
        element_cap=limits.max_payload_elements,
        minimum_value=(-1 if allow_border_sentinel else 0),
        maximum_value=v2.PAYLOAD_MAX_ABS_ELEMENT,
        sentinel_allowed=allow_border_sentinel)
    if counted != elements:
        raise SameAxisOwnershipPreflightError(
            REASON_SIDECAR_REPLAY_MISMATCH, "payload_array.values")
    try:
        array = np.asarray(values, dtype=np.dtype(dtype)).reshape(tuple(shape))
    except (TypeError, ValueError, OverflowError) as error:
        raise SameAxisOwnershipPreflightError(
            REASON_SIDECAR_REPLAY_MISMATCH, "payload_array",
            type(error).__name__) from error
    # Defense in depth: the exact-integer leaf and range validation above already
    # makes these post-allocation checks unreachable for an accepted array.
    if array.size:
        minimum = int(array.min())
        if minimum < (-1 if allow_border_sentinel else 0):
            raise SameAxisOwnershipPreflightError(
                REASON_SENTINEL_INVALID if allow_border_sentinel
                else REASON_TAP_INDEX_INVALID, "payload_array")
        if int(array.max()) > v2.PAYLOAD_MAX_ABS_ELEMENT:
            raise SameAxisOwnershipPreflightError(
                REASON_TAP_INDEX_INVALID, "payload_array")
    if hashlib.sha256(array.tobytes(order="C")).hexdigest() != record.get("sha256"):
        raise SameAxisOwnershipPreflightError(
            REASON_SIDECAR_REPLAY_MISMATCH, "payload_array_sha256")
    array.flags.writeable = False
    return array


def _payload_from_record(record: object, limits: PreflightLimitsV1) -> v1.MatcherSupportPayloadV1:
    record = _as_dict(record)
    try:
        return v1.MatcherSupportPayloadV1(**_init_kwargs(
            v1.MatcherSupportPayloadV1, record, rebuild={
                "logical_pairs": _array_from_record(
                    record["logical_pairs"], limits, allow_border_sentinel=False),
                "warp_source_rows": _array_from_record(
                    record["warp_source_rows"], limits, allow_border_sentinel=True),
                "warp_source_columns": _array_from_record(
                    record["warp_source_columns"], limits, allow_border_sentinel=True),
            }))
    except SameAxisOwnershipPreflightError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise SameAxisOwnershipPreflightError(
            REASON_SIDECAR_REPLAY_MISMATCH, "payload",
            type(error).__name__) from error


def _capture_from_record(capture_record: object,
                         limits: PreflightLimitsV1
                         ) -> tuple[v2.CompleteSupportCaptureV2, int]:
    capture_record = _as_dict(capture_record)
    if capture_record.get("schema") != v2.COMPLETE_SUPPORT_CAPTURE_SCHEMA:
        raise SameAxisOwnershipPreflightError(
            REASON_RECORD_MALFORMED, "capture.schema")
    allocated = 0

    def _index_page(record: object) -> v2.CandidateSupportIndexPageV2:
        record = _as_dict(record)
        try:
            rows = tuple(
                v2.CandidateSupportIndexRowV2(**_init_kwargs(
                    v2.CandidateSupportIndexRowV2, _as_dict(item),
                    rebuild={"cell_shift_rc": _optional_tuple(
                        _as_dict(item).get("cell_shift_rc"))}))
                for item in record.get("rows", ()))
            return v2.CandidateSupportIndexPageV2(**_init_kwargs(
                v2.CandidateSupportIndexPageV2, record, rebuild={"rows": rows}))
        except SameAxisOwnershipPreflightError:
            raise
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise SameAxisOwnershipPreflightError(
                REASON_INDEX_REPLAY_MISMATCH, "index_page",
                type(error).__name__) from error

    def _digest_page(record: object) -> v2.CandidateSupportDigestPageV2:
        try:
            return v2.CandidateSupportDigestPageV2(**_init_kwargs(
                v2.CandidateSupportDigestPageV2, record,
                rebuild={"digests": tuple(_as_dict(record).get("digests", ()))}))
        except SameAxisOwnershipPreflightError:
            raise
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise SameAxisOwnershipPreflightError(
                REASON_DIGEST_REPLAY_MISMATCH, "digest_page",
                type(error).__name__) from error

    def _sidecar_page(record: object) -> v2.MatcherSupportSidecarPageV2:
        nonlocal allocated
        entries = []
        for item in _as_dict(record).get("entries", ()):
            payload = _payload_from_record(_as_dict(item).get("payload"), limits)
            allocated += (int(payload.logical_pairs.size)
                          + int(payload.warp_source_rows.size)
                          + int(payload.warp_source_columns.size))
            if allocated > limits.max_payload_elements:
                raise SameAxisOwnershipPreflightError(
                    REASON_PAYLOAD_OVER_CAP, "payload_elements")
            try:
                entries.append(v2.MatcherSupportSidecarEntryV2(**_init_kwargs(
                    v2.MatcherSupportSidecarEntryV2, item,
                    rebuild={"payload": payload})))
            except SameAxisOwnershipPreflightError:
                raise
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise SameAxisOwnershipPreflightError(
                    REASON_SIDECAR_REPLAY_MISMATCH, "sidecar_entry",
                    type(error).__name__) from error
        try:
            return v2.MatcherSupportSidecarPageV2(**_init_kwargs(
                v2.MatcherSupportSidecarPageV2, record,
                rebuild={"entries": tuple(entries)}))
        except SameAxisOwnershipPreflightError:
            raise
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise SameAxisOwnershipPreflightError(
                REASON_SIDECAR_REPLAY_MISMATCH, "sidecar_page",
                type(error).__name__) from error

    def _capture_root(record: object) -> v2.CandidateSupportCaptureRootV2:
        record = _as_dict(record)
        plan_record = record.get("cap_plan")
        plan = (None if plan_record is None else v2.CompleteSupportCapPlanV1(
            **_init_kwargs(v2.CompleteSupportCapPlanV1, plan_record)))
        sections = tuple(v2.CandidateSupportDigestSectionV2(**_init_kwargs(
            v2.CandidateSupportDigestSectionV2, item, rebuild={"pages": tuple(
                v2.CandidateSupportDigestPageBindingV2(**_init_kwargs(
                    v2.CandidateSupportDigestPageBindingV2, binding))
                for binding in _as_dict(item).get("pages", ()))}))
            for item in record.get("digest_sections", ()))
        ingress = tuple(FrameIngressProvenanceV1.from_record(item)
                        for item in record.get("ingress", ()))
        root_record = {name: record.get(name)
                       for name in ("status", "basis", "reason", "field")}
        root_record.update({name: value for name, value in record.items()
                            if name not in ("status", "basis", "reason", "field")})
        return v2.CandidateSupportCaptureRootV2(**_init_kwargs(
            v2.CandidateSupportCaptureRootV2, root_record, rebuild={
                "index_pages": tuple(v2.CandidateSupportIndexPageBindingV2(
                    **_init_kwargs(v2.CandidateSupportIndexPageBindingV2, item))
                    for item in record["index_pages"]),
                "sidecar_pages": tuple(v2.CandidateSupportSidecarPageBindingV2(
                    **_init_kwargs(v2.CandidateSupportSidecarPageBindingV2, item))
                    for item in record["sidecar_pages"]),
                "digest_sections": sections,
                "ingress": ingress,
                "cap_plan": plan}))

    try:
        index_pages = tuple(_index_page(item)
                            for item in capture_record.get("index_pages", ()))
        digest_pages = tuple(_digest_page(item)
                             for item in capture_record.get("digest_pages", ()))
        sidecar_pages = tuple(_sidecar_page(item)
                              for item in capture_record.get("sidecar_pages", ()))
        root = _capture_root(capture_record["root"])
        capture = v2.CompleteSupportCaptureV2(
            root=root, index_pages=index_pages, digest_pages=digest_pages,
            sidecar_pages=sidecar_pages, schema=capture_record["schema"])
    except SameAxisOwnershipPreflightError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
        raise SameAxisOwnershipPreflightError(
            REASON_CAPTURE_VALIDATION_FAILED, "capture",
            type(error).__name__) from error
    try:
        v2.validate_complete_support_capture(capture)
    except SameAxisOwnershipPreflightError:
        raise
    except (ValueError, TypeError, AttributeError, KeyError, IndexError) as error:
        raise SameAxisOwnershipPreflightError(
            REASON_CAPTURE_VALIDATION_FAILED, "capture",
            type(error).__name__) from error
    return capture, allocated  # type: ignore[return-value]


# --- Wrapper pins -------------------------------------------------------------

def _verify_wrapper_pins(document: dict, capture: v2.CompleteSupportCaptureV2,
                         expectations: "FrozenRouteBExpectationsV1 | None") -> None:
    root = capture.root
    claimed_artifact = document.get("artifact_sha256")
    recomputed_artifact = _digest({key: value for key, value in document.items()
                                   if key != "artifact_sha256"})
    if not _hex64(claimed_artifact) or claimed_artifact != recomputed_artifact:
        raise SameAxisOwnershipPreflightError(
            REASON_ARTIFACT_TAMPER, "artifact_sha256")
    bindings = (
        ("record_sha256", root.record_sha256),
        ("whole_universe_identity_sha256", root.whole_universe_identity_sha256),
        ("input_match_identity_sha256", root.input_match_identity_sha256),
        ("failed_assignment_sha256", root.failed_assignment_sha256),
        ("expected_row_count", root.row_count),
    )
    for name, expected in bindings:
        if document.get(name) != expected:
            raise SameAxisOwnershipPreflightError(
                REASON_RECORD_PIN_MISMATCH, name)
    witnesses = _as_list(document.get("signature_witnesses"), "signature_witnesses")
    if document.get("signature_witness_count") != len(witnesses):
        raise SameAxisOwnershipPreflightError(
            REASON_RECORD_PIN_MISMATCH, "signature_witness_count")
    settings = _as_dict(document.get("settings"))
    if settings != {"max_shift_cells": 32, "integer_tolerance": 0.15,
                    "min_resultant": 0.9, "ambiguous_margin": 0.02}:
        raise SameAxisOwnershipPreflightError(
            REASON_RECORD_PIN_MISMATCH, "settings")
    stage = _as_dict(document.get("stage_digests"))
    if set(stage) != {"entry_sha256", "session_sha256", "lattice_sha256",
                      "signatures_sha256", "graph_sha256", "trace_sha256"}:
        raise SameAxisOwnershipPreflightError(
            REASON_RECORD_PIN_MISMATCH, "stage_digests")
    if any(not _hex64(value) for value in stage.values()):
        raise SameAxisOwnershipPreflightError(
            REASON_RECORD_PIN_MISMATCH, "stage_digests")
    if expectations is not None:
        pins = (
            ("artifact_sha256", expectations.artifact_sha256),
            ("record_sha256", expectations.record_sha256),
            ("whole_universe_identity_sha256",
             expectations.whole_universe_identity_sha256),
            ("input_match_identity_sha256",
             expectations.input_match_identity_sha256),
            ("failed_assignment_sha256", expectations.failed_assignment_sha256),
            ("expected_row_count", expectations.expected_row_count),
            ("signature_witness_count", expectations.signature_witness_count),
        )
        for name, expected in pins:
            if document.get(name) != expected:
                raise SameAxisOwnershipPreflightError(
                    REASON_RECORD_PIN_MISMATCH, f"frozen.{name}")


def _verify_ingress_records(document: dict,
                            capture: v2.CompleteSupportCaptureV2) -> tuple[tuple[str, ...], list[dict]]:
    """Cross-bind the four ingress views and return the ordered frame ids."""
    top_ingress = _as_list(document.get("ingress"), "ingress")
    inputs = _as_list(document.get("inputs"), "inputs")
    root_records = [record.to_record() for record in capture.root.ingress]
    if top_ingress != root_records:
        raise SameAxisOwnershipPreflightError(
            REASON_INGRESS_JOIN_MISMATCH, "record.ingress")
    expected_inputs = [{"frame_id": record["frame_id"], "path": record["source_label"],
                        "sha256": record["byte_sha256"]} for record in root_records]
    if inputs != expected_inputs:
        raise SameAxisOwnershipPreflightError(
            REASON_INGRESS_JOIN_MISMATCH, "record.inputs")
    frame_ids = [record.frame_id for record in capture.root.ingress]
    if len(set(frame_ids)) != len(frame_ids):
        raise SameAxisOwnershipPreflightError(
            REASON_DUPLICATE_INGRESS, "frame_id")
    for record in capture.root.ingress:
        resolved = record.transform.resolved_output_size
        if resolved != (record.decoded_shape[1], record.decoded_shape[0]):
            raise SameAxisOwnershipPreflightError(
                REASON_INGRESS_JOIN_MISMATCH, "transform.output_size")
    return tuple(frame_ids), [dict(record) for record in root_records]


# --- Signature-witness join ---------------------------------------------------

def _verify_hex64(value: object, field: str) -> str:
    if not _hex64(value):
        raise SameAxisOwnershipPreflightError(
            REASON_SIGNATURE_JOIN_MISMATCH, field)
    return value


def _parse_signature_witnesses(document: dict) -> tuple[dict, dict, dict]:
    witnesses = _as_list(document.get("signature_witnesses"), "signature_witnesses")
    bound: dict[str, dict] = {}
    refused: list[dict] = []
    seen_keys: set[tuple] = set()
    outcomes = {"bound": 0, "refused": 0}
    detail_counts: dict[str, int] = {}
    for index, item in enumerate(witnesses):
        item = _as_dict(item)
        outcome = item.get("outcome")
        if outcome not in ("bound", "refused"):
            raise SameAxisOwnershipPreflightError(
                REASON_SIGNATURE_JOIN_MISMATCH, f"witness.{index}.outcome")
        frame_id = item.get("frame_id")
        canonical_index = item.get("canonical_index")
        alternative_index = item.get("alternative_index")
        if (type(frame_id) is not str or not frame_id
                or type(canonical_index) is not int or canonical_index < 0
                or type(alternative_index) is not int or alternative_index < 0):
            raise SameAxisOwnershipPreflightError(
                REASON_SIGNATURE_JOIN_MISMATCH, f"witness.{index}")
        key = (frame_id, canonical_index, alternative_index)
        if key in seen_keys:
            raise SameAxisOwnershipPreflightError(
                REASON_DUPLICATE_WITNESS, f"witness.{index}")
        seen_keys.add(key)
        if outcome == "bound":
            signature = _verify_hex64(
                item.get("signature_sha256"), f"witness.{index}.signature_sha256")
            if signature in bound:
                raise SameAxisOwnershipPreflightError(
                    REASON_DUPLICATE_SIGNATURE, f"witness.{index}.signature_sha256")
            ingress_sha = _verify_hex64(
                item.get("ingress_sha256"), f"witness.{index}.ingress_sha256")
            for name in ("mask_sha256", "masks_sha256", "witness_sha256"):
                _verify_hex64(item.get(name), f"witness.{index}.{name}")
            for name in ("max_cells", "max_rectified_side", "pixels_per_cell"):
                if not _is_positive_int(item.get(name)):
                    raise SameAxisOwnershipPreflightError(
                        REASON_SIGNATURE_JOIN_MISMATCH, f"witness.{index}.{name}")
            if item.get("schema") != "mc-photogrammetry-source-tap-witness-binding-v1":
                raise SameAxisOwnershipPreflightError(
                    REASON_SIGNATURE_JOIN_MISMATCH, f"witness.{index}.schema")
            record = dict(item)
            record["canonical_index"] = canonical_index
            record["alternative_index"] = alternative_index
            bound[signature] = record
            outcomes["bound"] += 1
        else:
            reason = item.get("reason")
            if reason not in (v1.SOURCE_TAP_WITNESS_BUDGET_EXHAUSTED,
                              v1.SOURCE_TAP_WITNESS_UNAVAILABLE,
                              v1.SOURCE_TAP_WITNESS_SHAPE_MISMATCH,
                              v1.SOURCE_TAP_WITNESS_INDEX_OUT_OF_RANGE,
                              v1.INGRESS_PROVENANCE_UNAVAILABLE):
                raise SameAxisOwnershipPreflightError(
                    REASON_SIGNATURE_JOIN_MISMATCH, f"witness.{index}.reason")
            detail = item.get("detail", "")
            detail_counts[str(detail)] = detail_counts.get(str(detail), 0) + 1
            refused.append(dict(item))
            outcomes["refused"] += 1
    facts = {
        "bound_count": outcomes["bound"],
        "refused_count": outcomes["refused"],
        "refusal_detail_counts": detail_counts,
        "witness_count": len(witnesses),
    }
    return bound, facts, {"refused": refused}


# --- Endpoint / alternative join ----------------------------------------------

def verify_endpoint_alternative_join(
    rows: Sequence[object],
    sections: Mapping[str, Sequence[object]],
    bound_by_signature: Mapping[str, object],
    *,
    sidecar_entry_max_raw_bytes: int = SIDECAR_ENTRY_MAX_RAW_BYTES,
) -> EndpointJoinFactsV1:
    """Join each admitted row's left/right signature digest to its witness.

    ``bound_by_signature`` maps ``signature_sha256`` to a witness binding with
    ``frame_id`` and ``alternative_index``. The left endpoint must bind the
    row's ``left`` frame and ``left_alternative_index`` and the right endpoint
    the ``right`` pair; an asymmetric (swapped) source/target orientation is a
    typed refusal. Unbound endpoints are counted as witness-availability gaps,
    never as incorrect rows.
    """
    left_section = sections.get("left_signature_sha256", ())
    right_section = sections.get("right_signature_sha256", ())
    if len(left_section) != len(rows) or len(right_section) != len(rows):
        raise SameAxisOwnershipPreflightError(
            REASON_SIGNATURE_JOIN_MISMATCH, "signature_sections")
    referenced: set[str] = set()
    missing_rows = 0
    missing_over_cap = 0
    missing_refusal = 0
    raw_cap_provable = 0
    for position, row in enumerate(rows):
        left_digest = left_section[position]
        right_digest = right_section[position]
        unbound = False
        for endpoint, digest, frame, alternative in (
                ("left", left_digest, row.left, row.left_alternative_index),
                ("right", right_digest, row.right, row.right_alternative_index)):
            if digest is None:
                unbound = True
                continue
            if not _hex64(digest):
                raise SameAxisOwnershipPreflightError(
                    REASON_SIGNATURE_JOIN_MISMATCH, f"{endpoint}_signature")
            referenced.add(digest)
            witness = bound_by_signature.get(digest)
            if witness is None:
                unbound = True
                continue
            if isinstance(witness, Mapping):
                witness_frame = witness.get("frame_id")
                witness_alternative = witness.get("alternative_index")
            else:
                witness_frame = getattr(witness, "frame_id", None)
                witness_alternative = getattr(witness, "alternative_index", None)
            if witness_frame != frame or witness_alternative != alternative:
                raise SameAxisOwnershipPreflightError(
                    REASON_ENDPOINT_ALTERNATIVE_MISMATCH,
                    f"row.{position}.{endpoint}")
        if unbound:
            missing_rows += 1
            if row.support_refusal_reason == "support_payload_over_cap":
                missing_over_cap += 1
            elif row.support_refusal_reason == "source_tap_witness_unavailable":
                missing_refusal += 1
        if (row.support_refusal_reason == "support_payload_over_cap"
                and 32 * int(row.overlap_cells) > sidecar_entry_max_raw_bytes):
            raw_cap_provable += 1
    bound_keys = set(bound_by_signature)
    referenced_bound = len(referenced & bound_keys)
    referenced_unbound = len(referenced - bound_keys)
    bound_unreferenced = len(bound_keys - referenced)
    return EndpointJoinFactsV1(
        endpoint_rows=len(rows),
        referenced_signature_count=len(referenced),
        referenced_bound_count=referenced_bound,
        referenced_unbound_count=referenced_unbound,
        bound_unreferenced_count=bound_unreferenced,
        missing_witness_row_count=missing_rows,
        missing_witness_over_cap_count=missing_over_cap,
        missing_witness_refusal_count=missing_refusal,
        raw_cap_provable_count=raw_cap_provable)


def verify_ingress_join(
    ingress_records: Sequence[Mapping[str, object]],
    bound_by_signature: Mapping[str, Mapping[str, object]],
) -> dict:
    """Bind every bound witness ``ingress_sha256`` to one ordered ingress pin."""
    by_hash: dict[str, str] = {}
    for record in ingress_records:
        digest = _digest(record)
        frame_id = record.get("frame_id")
        if type(frame_id) is not str or not frame_id:
            raise SameAxisOwnershipPreflightError(
                REASON_INGRESS_JOIN_MISMATCH, "ingress.frame_id")
        if digest in by_hash:
            raise SameAxisOwnershipPreflightError(
                REASON_DUPLICATE_INGRESS, "ingress_sha256")
        by_hash[digest] = frame_id
    bound_count = 0
    for signature, witness in bound_by_signature.items():
        ingress_sha = witness.get("ingress_sha256")
        frame_id = witness.get("frame_id")
        if ingress_sha not in by_hash:
            raise SameAxisOwnershipPreflightError(
                REASON_INGRESS_JOIN_MISMATCH, f"witness.{signature[:12]}")
        if by_hash[ingress_sha] != frame_id:
            raise SameAxisOwnershipPreflightError(
                REASON_INGRESS_JOIN_MISMATCH, f"witness.{signature[:12]}.frame_id")
        bound_count += 1
    return {"ingress_count": len(ingress_records), "bound_ingress_count": bound_count,
            "ingress_sha256_values": tuple(sorted(by_hash))}


# --- Source-tap surface (sidecar warp maps) -----------------------------------

def verify_source_tap_index_semantics(values: Sequence[int], source_cells: int,
                                      *, allow_border_sentinel: bool = True) -> int:
    """Validate flat source-tap indices and return the count of ``-1`` sentinels.

    ``-1`` is the documented border sentinel for a tap outside the decoded
    source frame; ``0`` is a real index. Anything below ``-1`` or at/above
    ``source_cells`` fails closed.
    """
    if type(source_cells) is not int or source_cells <= 0:
        raise SameAxisOwnershipPreflightError(
            REASON_TAP_INDEX_INVALID, "source_cells")
    sentinels = 0
    for raw in values:
        if type(raw) is not int:
            raise SameAxisOwnershipPreflightError(
                REASON_TAP_INDEX_INVALID, "tap_value")
        if raw == -1:
            if not allow_border_sentinel:
                raise SameAxisOwnershipPreflightError(
                    REASON_TAP_INDEX_INVALID, "tap_value")
            sentinels += 1
        elif raw < -1 or raw >= source_cells:
            raise SameAxisOwnershipPreflightError(
                REASON_SENTINEL_INVALID if raw < -1 else REASON_TAP_INDEX_INVALID,
                "tap_value")
    return sentinels


def _count_tap_sentinels(array: "object", source_cells: int) -> int:
    """Fast NumPy form of :func:`verify_source_tap_index_semantics`."""
    if type(source_cells) is not int or source_cells <= 0:
        raise SameAxisOwnershipPreflightError(
            REASON_TAP_INDEX_INVALID, "source_cells")
    if array.size:
        minimum = int(array.min())
        if minimum < -1:
            raise SameAxisOwnershipPreflightError(
                REASON_SENTINEL_INVALID, "tap_value")
        if int(array.max()) >= source_cells:
            raise SameAxisOwnershipPreflightError(
                REASON_TAP_INDEX_INVALID, "tap_value")
        return int((array == -1).sum())
    return 0


def _verify_tap_surface(rows: Sequence[object],
                        sidecar_entries: Sequence[object]) -> TapSurfaceFactsV1:
    present = [row for row in rows if row.support_state == v2.SUPPORT_STATE_PRESENT]
    by_ordinal: dict[int, object] = {}
    for entry in sidecar_entries:
        ordinal = entry.admitted_row_index
        if ordinal in by_ordinal:
            raise SameAxisOwnershipPreflightError(
                REASON_SIDECAR_REPLAY_MISMATCH, "sidecar.duplicate_entry")
        by_ordinal[ordinal] = entry
    present_ordinals = [row.admitted_row_index for row in present]
    if sorted(by_ordinal) != present_ordinals:
        missing = sorted(set(present_ordinals) - set(by_ordinal))
        if missing:
            raise SameAxisOwnershipPreflightError(
                REASON_SOURCE_TAP_RECONSTRUCTION_UNAVAILABLE,
                f"sidecar.entry.{missing[0]}")
        raise SameAxisOwnershipPreflightError(
            REASON_SIDECAR_REPLAY_MISMATCH, "sidecar.present_rows")
    payload_elements = 0
    sentinel_count = 0
    logical_pairs = 0
    for row in present:
        entry = by_ordinal[row.admitted_row_index]
        if entry.payload_sha256 != row.support_payload_sha256:
            raise SameAxisOwnershipPreflightError(
                REASON_ENVELOPE_JOIN_MISMATCH,
                f"row.{row.admitted_row_index}.support_payload_sha256")
        if int(entry.overlap_cells) != int(row.overlap_cells):
            raise SameAxisOwnershipPreflightError(
                REASON_ENVELOPE_JOIN_MISMATCH,
                f"row.{row.admitted_row_index}.overlap_cells")
        payload = entry.payload
        if tuple(entry.source_shape) != tuple(payload.source_shape):
            raise SameAxisOwnershipPreflightError(
                REASON_ENVELOPE_JOIN_MISMATCH,
                f"row.{row.admitted_row_index}.source_shape")
        if tuple(entry.target_shape) != tuple(payload.target_shape):
            raise SameAxisOwnershipPreflightError(
                REASON_ENVELOPE_JOIN_MISMATCH,
                f"row.{row.admitted_row_index}.target_shape")
        if len(payload.logical_pairs) != int(row.overlap_cells):
            raise SameAxisOwnershipPreflightError(
                REASON_ENVELOPE_JOIN_MISMATCH,
                f"row.{row.admitted_row_index}.logical_pairs")
        source_cells = int(payload.source_shape[0]) * int(payload.source_shape[1])
        sentinel_count += _count_tap_sentinels(payload.warp_source_rows, source_cells)
        sentinel_count += _count_tap_sentinels(
            payload.warp_source_columns, source_cells)
        logical_pairs += int(payload.logical_pairs.size)
        payload_elements += (int(payload.logical_pairs.size)
                             + int(payload.warp_source_rows.size)
                             + int(payload.warp_source_columns.size))
    return TapSurfaceFactsV1(
        present_rows=len(present), sidecar_entries=len(sidecar_entries),
        payload_elements=payload_elements,
        border_sentinel_count=sentinel_count, logical_pair_count=logical_pairs)


# --- Cross-tabs and pins ------------------------------------------------------

def _cross_tabs(rows: Sequence[object]) -> tuple[tuple[str, str, "str | None", int], ...]:
    # The canonical table order is the accepted decomposition order: kept then
    # dropped, present then refused, and over-cap then witness-unavailable.
    disposition_order = {plane_graph.SAME_AXIS_DISPOSITION_KEPT: 0,
                         plane_graph.SAME_AXIS_DISPOSITION_DROPPED: 1}
    support_order = {v2.SUPPORT_STATE_PRESENT: 0, v2.SUPPORT_STATE_REFUSED: 1}
    refusal_order = {None: 0, "support_payload_over_cap": 1,
                     "source_tap_witness_unavailable": 2}
    counts: dict[tuple[str, str, "str | None"], int] = {}
    for row in rows:
        key = (row.disposition, row.support_state, row.support_refusal_reason)
        counts[key] = counts.get(key, 0) + 1

    def order(item: tuple) -> tuple:
        key = item[0]
        return (disposition_order.get(key[0], 9), support_order.get(key[1], 9),
                refusal_order.get(key[2], 5), str(key[2] or ""))

    return tuple((key[0], key[1], key[2], count)
                 for key, count in sorted(counts.items(), key=order))


def _frame_pin_inventory(ingress_records: Sequence[FrameIngressProvenanceV1]
                         ) -> tuple[FrameAlignmentPinInventoryV1, ...]:
    inventories = []
    for record in ingress_records:
        transform = record.transform
        transform_digest = _digest(record.to_record())
        pins = (
            PinStateV1(PIN_SOURCE_TRANSFORM, PIN_STATE_AVAILABLE,
                       source="record.ingress.transform",
                       detail=("source_size=%s crop_origin=%s crop_size=%s "
                               "output_size=%s resolved_output_size=%s ingress=%s"
                               % (list(transform.source_size),
                                  list(transform.crop_origin),
                                  (None if transform.crop_size is None
                                   else list(transform.crop_size)),
                                  (None if transform.output_size is None
                                   else list(transform.output_size)),
                                  list(transform.resolved_output_size),
                                  transform_digest[:16]))),
            PinStateV1(PIN_CAMERA, PIN_STATE_UNAVAILABLE,
                       detail="camera_pin_absent_from_frozen_record"),
            PinStateV1(PIN_INTRINSICS, PIN_STATE_UNAVAILABLE,
                       detail="intrinsics_pin_absent_from_frozen_record"),
            PinStateV1(PIN_POSE, PIN_STATE_UNAVAILABLE,
                       detail="pose_pin_absent_from_frozen_record"),
            PinStateV1(PIN_OCCLUSION, PIN_STATE_UNAVAILABLE,
                       detail="occlusion_convention_absent_from_frozen_record"),
            PinStateV1(PIN_EVALUATOR_DOMAIN, PIN_STATE_UNAVAILABLE,
                       detail="evaluator_domain_absent_from_frozen_record"),
        )
        inventories.append(FrameAlignmentPinInventoryV1(
            frame_id=record.frame_id, pins=pins))
    return tuple(inventories)


def _check_expectations(result: SameAxisOwnershipPreflightV1, facts: dict,
                        expectations: FrozenRouteBExpectationsV1) -> None:
    for name, expected in (
            ("row_count", expectations.expected_row_count),
            ("present_row_count", expectations.present_row_count),
            ("unknown_row_count", expectations.unknown_row_count),
            ("kept_row_count", expectations.kept_row_count),
            ("dropped_row_count", expectations.dropped_row_count),
            ("group_count", expectations.group_count),
            ("index_page_count", expectations.index_page_count),
            ("digest_page_count", expectations.digest_page_count),
            ("sidecar_page_count", expectations.sidecar_page_count),
            ("sidecar_entry_count", expectations.sidecar_entry_count),
            ("signature_witness_count", expectations.signature_witness_count),
            ("bound_witness_count", expectations.bound_witness_count),
            ("refused_witness_count", expectations.refused_witness_count)):
        if facts.get(name) != expected:
            raise SameAxisOwnershipPreflightError(
                REASON_CROSS_TAB_MISMATCH, f"frozen.{name}")
    if tuple(result.cross_tabs) != tuple(expectations.cross_tabs):
        raise SameAxisOwnershipPreflightError(
            REASON_CROSS_TAB_MISMATCH, "cross_tabs")
    endpoint = result.endpoint_join
    if endpoint is None:
        raise SameAxisOwnershipPreflightError(
            REASON_CROSS_TAB_MISMATCH, "endpoint_join")
    for name, expected in (
            ("referenced_signature_count", expectations.referenced_signature_count),
            ("referenced_bound_count", expectations.referenced_bound_count),
            ("referenced_unbound_count", expectations.referenced_unbound_count),
            ("bound_unreferenced_count", expectations.bound_unreferenced_count),
            ("missing_witness_row_count", expectations.missing_witness_row_count),
            ("missing_witness_over_cap_count",
             expectations.missing_witness_over_cap_count),
            ("missing_witness_refusal_count",
             expectations.missing_witness_refusal_count),
            ("raw_cap_provable_count", expectations.raw_cap_provable_count)):
        if getattr(endpoint, name) != expected:
            raise SameAxisOwnershipPreflightError(
                REASON_CROSS_TAB_MISMATCH, f"frozen.{name}")
    if tuple(result.ingress_frame_ids) != expectations.ingress_frame_ids:
        raise SameAxisOwnershipPreflightError(
            REASON_CROSS_TAB_MISMATCH, "frozen.ingress_frame_ids")


# --- Orchestration ------------------------------------------------------------

def _decode_sampling_section(document: dict,
                             capture: v2.CompleteSupportCaptureV2,
                             ) -> tuple[object, dict]:
    """Validate and decode the V2 sampling section against typed ingress pins.

    The ordered outcomes are cross-bound to the record's own witness summary
    first, so a foreign or swapped identity, a missing/duplicate entry or a
    modified array whose nested digest no longer matches refuses here. Only then
    are the bounded arrays decoded and every bound witness rebuilt (with its
    ingress ``ingress_sha256`` required to match the pin). No ownership score is
    computed; the summary carries counts, byte accounting and the section digest
    only.
    """
    section = document.get("sampling_persistence")
    try:
        persistence.cross_bind_witness_summaries(
            section, document.get("signature_witnesses"))
        ingress = {pin.frame_id: pin for pin in capture.root.ingress}
        # Root-pinned ingress is required for every bound witness before the
        # section can be labeled available: a bound entry whose frame has no
        # capture-root pin is a typed refusal here rather than a silently
        # unbound outcome, and the binding constructor additionally re-checks
        # each rectification ``source_shape`` against the pinned frame shape.
        decoded = persistence.decode_section(
            section, ingress_by_frame=ingress, require_ingress=True)
    except persistence.SamplingPersistenceError as error:
        raise SameAxisOwnershipPreflightError(
            REASON_SAMPLING_PERSISTENCE_INVALID, error.field,
            error.reason) from error
    ingress_bound = sum(1 for entry in decoded.entries
                        if entry.outcome == persistence.SAMPLING_PERSISTENCE_OUTCOME_BOUND
                        and entry.binding is not None)
    ingress_unbound = sum(1 for entry in decoded.entries
                          if entry.outcome == persistence.SAMPLING_PERSISTENCE_OUTCOME_BOUND
                          and entry.binding is None)
    record = {
        "schema": decoded.schema,
        "status": decoded.status,
        "reason": decoded.reason,
        "entry_count": decoded.entry_count,
        "bound_count": decoded.bound_count,
        "refused_count": decoded.refused_count,
        "persistence_refused_count": decoded.persistence_refused_count,
        "ingress_bound_count": ingress_bound,
        "ingress_unbound_count": ingress_unbound,
        "payload_bytes": decoded.payload_bytes,
        "metadata_bytes": decoded.metadata_bytes,
        "work_units": decoded.work_units,
        "section_sha256": decoded.section_sha256,
        "state": decoded.state,
    }
    return decoded, record


def _complete_result(document: dict, bounded: BoundedRecordBytesV1, prescan: dict,
                     capture: v2.CompleteSupportCaptureV2,
                     allocated: int,
                     expectations: "FrozenRouteBExpectationsV1 | None",
                     gate: InferenceDigestGate) -> SameAxisOwnershipPreflightV1:
    rows = tuple(row for page in capture.index_pages for row in page.rows)
    sections = {page.section: page.digests for page in capture.digest_pages}
    sidecar_entries = tuple(entry for page in capture.sidecar_pages
                            for entry in page.entries)
    bound, witness_facts, _refused = _parse_signature_witnesses(document)
    # Repeated sections (should be one page per section) are concatenated in the
    # page order the validator already accepted.
    section_values: dict[str, tuple] = {}
    for section in v2.DIGEST_SECTIONS:
        values: list = []
        for page in capture.digest_pages:
            if page.section == section:
                values.extend(page.digests)
        section_values[section] = tuple(values)
    endpoint = verify_endpoint_alternative_join(rows, section_values, bound)
    ingress_frame_ids, ingress_records = _verify_ingress_records(document, capture)
    ingress_facts = verify_ingress_join(ingress_records, bound)
    if ingress_facts["ingress_count"] != len(ingress_frame_ids):
        raise SameAxisOwnershipPreflightError(
            REASON_INGRESS_JOIN_MISMATCH, "ingress")
    taps = _verify_tap_surface(rows, sidecar_entries)
    # Cross-check the plain-document prescan against the typed rebuild: the two
    # independent passes must agree on every count they both measured.
    if (len(rows) != prescan["row_count"]
            or len(sidecar_entries) != prescan["sidecar_entry_count"]
            or taps.payload_elements != prescan["payload_elements"]):
        raise SameAxisOwnershipPreflightError(
            REASON_IDENTITY_JOIN_MISMATCH, "prescan_rebuild")
    cross_tabs = _cross_tabs(rows)
    present = sum(1 for row in rows if row.support_state == v2.SUPPORT_STATE_PRESENT)
    unknown = sum(1 for row in rows if row.support_state == v2.SUPPORT_STATE_REFUSED)
    kept = sum(1 for row in rows
               if row.disposition == plane_graph.SAME_AXIS_DISPOSITION_KEPT)
    dropped = len(rows) - kept
    group_count = len({row.group_index for row in rows})

    # The frozen inference digest is complete and replayed; only now may the
    # evaluator-domain inventory be labeled.
    gate.seal_inference_digest(
        artifact_sha256=document["artifact_sha256"],
        record_sha256=document["record_sha256"],
        whole_universe_identity_sha256=document["whole_universe_identity_sha256"],
        input_match_identity_sha256=document["input_match_identity_sha256"],
        failed_assignment_sha256=document["failed_assignment_sha256"])
    gate.require_evaluator_access("frame_pins")
    frame_pins = _frame_pin_inventory(capture.root.ingress)

    blockers = (
        "sampling_witness_tap_tables_not_persisted",
        "camera_pin_absent", "intrinsics_pin_absent", "pose_pin_absent",
        "occlusion_convention_absent", "evaluator_domain_absent",
    )
    sampling_witness_state = PIN_STATE_UNAVAILABLE
    sampling_witness_seam = SAMPLING_WITNESS_TAP_SEAM
    sampling_persistence_record = None
    if document.get("schema") == RECORD_SCHEMA_V2:
        decoded, sampling_persistence_record = _decode_sampling_section(
            document, capture)
        if decoded.entry_count != prescan["sampling_entry_count"]:
            raise SameAxisOwnershipPreflightError(
                REASON_SAMPLING_PERSISTENCE_INVALID, "sampling_persistence.join")
        if decoded.state == persistence.SAMPLING_PERSISTENCE_STATE_AVAILABLE:
            sampling_witness_state = PIN_STATE_AVAILABLE
            sampling_witness_seam = ""
            blockers = tuple(
                name for name in blockers
                if name != "sampling_witness_tap_tables_not_persisted")
        else:
            sampling_witness_seam = (
                SAMPLING_WITNESS_SECTION_INVALID
                if not decoded.reason else
                f"{SAMPLING_WITNESS_SECTION_INVALID}: {decoded.reason}")
    result = SameAxisOwnershipPreflightV1(
        status=STATUS_COMPLETE,
        input_bytes=bounded.byte_length, input_sha256=bounded.sha256,
        max_json_depth=bounded.max_json_depth, json_tokens=bounded.json_tokens,
        record_schema=document["schema"], record_status=document["status"],
        row_count=len(rows), present_row_count=present, unknown_row_count=unknown,
        kept_row_count=kept, dropped_row_count=dropped, group_count=group_count,
        index_page_count=capture.root.index_page_count,
        digest_page_count=capture.root.digest_page_count,
        sidecar_page_count=capture.root.sidecar_page_count,
        sidecar_entry_count=len(sidecar_entries),
        payload_element_count=taps.payload_elements,
        payload_elements_allocated=allocated,
        signature_witness_count=witness_facts["witness_count"],
        bound_witness_count=witness_facts["bound_count"],
        refused_witness_count=witness_facts["refused_count"],
        cross_tabs=cross_tabs, endpoint_join=endpoint,
        ingress_frame_ids=ingress_frame_ids, frame_pins=frame_pins,
        tap_surface_state=PIN_STATE_AVAILABLE,
        sampling_witness_state=sampling_witness_state,
        sampling_witness_seam=sampling_witness_seam,
        sampling_persistence=sampling_persistence_record,
        ownership_authorization=OWNERSHIP_WITHHELD,
        ownership_blockers=blockers, gate_sealed=gate.sealed,
        evaluator_payload_read=False)
    facts = {
        "row_count": result.row_count,
        "present_row_count": result.present_row_count,
        "unknown_row_count": result.unknown_row_count,
        "kept_row_count": result.kept_row_count,
        "dropped_row_count": result.dropped_row_count,
        "group_count": result.group_count,
        "index_page_count": result.index_page_count,
        "digest_page_count": result.digest_page_count,
        "sidecar_page_count": result.sidecar_page_count,
        "sidecar_entry_count": result.sidecar_entry_count,
        "signature_witness_count": result.signature_witness_count,
        "bound_witness_count": result.bound_witness_count,
        "refused_witness_count": result.refused_witness_count,
    }
    if expectations is not None:
        _check_expectations(result, facts, expectations)
    return result


def run_same_axis_ownership_preflight(
    record_path: "str | os.PathLike[str] | None" = None,
    *,
    limits: PreflightLimitsV1 = PreflightLimitsV1(),
    expectations: "FrozenRouteBExpectationsV1 | None" = FROZEN_ROUTE_B_EXPECTATIONS,
    reader: Callable[[object], bytes] | None = None,
    clock: Callable[[], float] = time.monotonic,
    gate: InferenceDigestGate | None = None,
) -> SameAxisOwnershipPreflightV1:
    """Run the bounded, fail-closed frozen V2 support/ownership preflight.

    Never raises for a record-level refusal: a typed ``refused`` result is
    returned instead, with zero counts so no unverified statistic is emitted.
    """
    path = frozen_record_path() if record_path is None else record_path
    bounded: BoundedRecordBytesV1 | None = None
    detected_schema = ""
    gate = InferenceDigestGate() if gate is None else gate
    try:
        deadline = _Deadline(limits.max_wall_seconds, clock)
        bounded = read_bounded_record_bytes(path, limits, reader=reader, clock=clock)
        if expectations is not None:
            if bounded.byte_length != expectations.input_bytes:
                raise SameAxisOwnershipPreflightError(
                    REASON_RECORD_PIN_MISMATCH, "frozen.input_bytes")
            if bounded.sha256 != expectations.input_sha256:
                raise SameAxisOwnershipPreflightError(
                    REASON_ARTIFACT_TAMPER, "input_sha256")
        deadline.check()
        try:
            document = json.loads(bounded.data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SameAxisOwnershipPreflightError(
                REASON_RECORD_MALFORMED, "record.json",
                type(error).__name__) from error
        if type(document) is not dict:
            raise SameAxisOwnershipPreflightError(
                REASON_RECORD_MALFORMED, "record.json")
        detected_schema = str(document.get("schema", ""))
        deadline.check()
        # Cap/duplicate prescan on the plain document: no typed record, no
        # NumPy array and no evaluator access has happened yet.
        prescan = _prescan_structure(document, limits)
        deadline.check()
        capture, allocated = _capture_from_record(document.get("capture"), limits)
        deadline.check()
        _verify_wrapper_pins(document, capture, expectations)
        deadline.check()
        result = _complete_result(document, bounded, prescan, capture, allocated,
                                  expectations, gate)
        deadline.check()
        return result
    except SameAxisOwnershipPreflightError as error:
        return _refused_result(error.reason, error.field, error.detail,
                               bounded=bounded,
                               record_schema=(detected_schema or
                                              ("" if bounded is None
                                               else RECORD_SCHEMA_V1)),
                               record_status="")


# --- Result JSON helper -------------------------------------------------------

def result_record(result: SameAxisOwnershipPreflightV1) -> dict:
    """The compact JSON result body (no rows, no ownership percentage)."""
    return result.to_record()


__all__ = [
    "PREFLIGHT_SCHEMA", "STATUS_COMPLETE", "STATUS_REFUSED",
    "MAX_INPUT_BYTES", "MAX_JSON_DEPTH", "MAX_JSON_TOKENS",
    "MAX_UNIVERSE_ROWS", "MAX_SIGNATURE_WITNESSES", "MAX_INDEX_PAGES",
    "MAX_DIGEST_PAGES", "MAX_SIDECAR_PAGES", "MAX_PAYLOAD_ELEMENTS",
    "MAX_PAYLOAD_BYTES", "PAYLOAD_CONTRACT_DTYPE",
    "PAYLOAD_CONTRACT_ITEM_BYTES",
    "MAX_INGRESS_RECORDS", "MAX_WALL_SECONDS", "SIDECAR_ENTRY_MAX_RAW_BYTES",
    "REQUIRED_ALIGNMENT_PINS", "PIN_STATE_AVAILABLE", "PIN_STATE_UNAVAILABLE",
    "SAMPLING_WITNESS_TAP_SEAM", "OWNERSHIP_WITHHELD",
    "REFUSAL_REASONS",
    "REASON_INPUT_OVER_CAP", "REASON_INPUT_UNREADABLE",
    "REASON_INPUT_SIZE_DRIFT", "REASON_DEPTH_OVER_CAP", "REASON_WORK_OVER_CAP",
    "REASON_TIME_OVER_CAP", "REASON_RECORD_MALFORMED",
    "REASON_RECORD_PIN_MISMATCH", "REASON_ARTIFACT_TAMPER",
    "REASON_ROW_CAP_EXCEEDED", "REASON_PAGE_CAP_EXCEEDED",
    "REASON_WITNESS_CAP_EXCEEDED", "REASON_PAYLOAD_OVER_CAP",
    "REASON_PAYLOAD_DTYPE_INVALID", "REASON_PAYLOAD_VALUE_INVALID",
    "REASON_DUPLICATE_ROW_IDENTITY", "REASON_DUPLICATE_SIGNATURE",
    "REASON_DUPLICATE_WITNESS", "REASON_DUPLICATE_INGRESS",
    "REASON_ORIENTATION_MISMATCH", "REASON_INDEX_REPLAY_MISMATCH",
    "REASON_DIGEST_REPLAY_MISMATCH", "REASON_SIDECAR_REPLAY_MISMATCH",
    "REASON_CAPTURE_VALIDATION_FAILED", "REASON_IDENTITY_JOIN_MISMATCH",
    "REASON_ENVELOPE_JOIN_MISMATCH", "REASON_ENDPOINT_ALTERNATIVE_MISMATCH",
    "REASON_SIGNATURE_JOIN_MISMATCH", "REASON_INGRESS_JOIN_MISMATCH",
    "REASON_SOURCE_TAP_RECONSTRUCTION_UNAVAILABLE", "REASON_SENTINEL_INVALID",
    "REASON_TAP_INDEX_INVALID", "REASON_CROSS_TAB_MISMATCH",
    "REASON_SAMPLING_PERSISTENCE_INVALID",
    "REASON_EVALUATOR_ACCESS_BEFORE_INFERENCE_DIGEST",
    "RECORD_SCHEMA_V1", "RECORD_SCHEMA_V2", "ACCEPTED_RECORD_SCHEMAS",
    "SAMPLING_WITNESS_SECTION_INVALID",
    "SameAxisOwnershipPreflightError", "EvaluatorAccessRefused",
    "PreflightLimitsV1", "FrozenRouteBExpectationsV1",
    "FROZEN_ROUTE_B_EXPECTATIONS", "frozen_record_path",
    "BoundedRecordBytesV1", "scan_json_bounds", "read_bounded_record_bytes",
    "PinStateV1", "FrameAlignmentPinInventoryV1", "EndpointJoinFactsV1",
    "TapSurfaceFactsV1", "InferenceDigestSealV1", "InferenceDigestGate",
    "SameAxisOwnershipPreflightV1",
    "verify_endpoint_alternative_join", "verify_ingress_join",
    "verify_source_tap_index_semantics", "run_same_axis_ownership_preflight",
    "result_record",
]
