"""Additive, default-off V2 complete-support contract and synthetic preflight.

This module is the bounded implementation of the accepted design
``docs/reports/DS-R4-COMPLETE-SUPPORT-DESIGN-2026-10-04.md``. It is a **single
additive sibling** of :mod:`candidate_support_provenance` (the shipped V1
module) and changes no V1 file, constant, record, digest or default code path.
Nothing here runs on the default production path: there is no graph sink
plumbing, no capture, no evaluator/truth/camera/mask/image read, and the
preflight is pure synthetic arithmetic over declared synthetic shapes.

What it owns:

* :class:`CandidateSupportIndexRowV2` / :class:`CandidateSupportIndexPageV2` -
  the complete per-admitted-candidate disposition index (all rows, kept or
  dropped), with row-local typed support state and no inline arrays;
* :class:`MatcherSupportSidecarEntryV2` / :class:`MatcherSupportSidecarPageV2` -
  the content-bound support sidecar (arrays isolated from the index);
* :class:`CandidateSupportDigestPageV2` - the six bounded ordered digest
  sections, with explicit ``None`` at a position whose material is unavailable;
* :class:`CandidateSupportCaptureRootV2` / :class:`CompleteSupportCaptureV2` -
  the small binding root plus the staged, all-or-nothing capture;
* :func:`build_complete_support_capture` - pure assembly from pre-built rows,
  digests and sidecar entries (used by the focused tests);
* :func:`collect_complete_candidate_support` - the sink adapter over the same
  five inputs V1 consumes, applying row-local typed support refusal;
* :func:`preflight_complete_support` - the deterministic synthetic worst-case
  ceiling preflight that measures every canonical size and proposes caps;
* :data:`ROUTE_B_DIAGNOSTIC_CAP_PLAN` / :func:`collect_route_b_diagnostic_support`
  - the exact single-run cap profile the lead accepted for the frozen 1,928-row
  ordinary-image Route B diagnostic, and the default-off collector that the
  graph's ``complete_support_sink`` seam calls with it.

Identity decisions implemented exactly as the Lead fixed them:

* unavailable digest slots are JSON ``null`` at their ordered position;
* a missing row-local witness/support is a typed row refusal that preserves the
  disposition and leaves the capture ``complete``;
* ``complete_support_universe`` is a new domain over **all** admitted ordered
  rows, while the kept-row ``input_match_identity`` digest is unchanged.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field as dataclass_field

import numpy as np

from mc_photogrammetry.pipeline.contracts import FrameTransform
from mc_photogrammetry.pipeline.candidate_support_provenance import (
    CANDIDATE_SUPPORT_ORIENTATION,
    CANDIDATE_SUPPORT_TOLERANCE,
    MATCHER_WITNESS_UNAVAILABLE,
    SUPPORT_PAYLOAD_OVER_CAP,
    SUPPORT_PAYLOAD_UNAVAILABLE,
    CandidateSupportError,
    FrameIngressProvenanceV1,
    MatcherSupportPayloadV1,
    SourceTapWitnessBindingV1,
    _alternative_index,
    _delta,
    _hex64,
    _identity_digest,
    _sha,
    _support_payload,
)
from mc_photogrammetry.reconstruction import plane_graph

__all__ = [
    "COMPLETE_SUPPORT_SCHEMA",
    "COMPLETE_SUPPORT_BASIS",
    "COMPLETE_SUPPORT_DIGEST_DOMAIN",
    "COMPLETE_SUPPORT_UNIVERSE_DOMAIN",
    "COMPLETE_SUPPORT_UNIVERSE_SCHEMA",
    "COMPLETE_SUPPORT_CAPTURE_SCHEMA",
    "INDEX_PAGE_SCHEMA",
    "SIDECAR_ENTRY_SCHEMA",
    "SIDECAR_PAGE_SCHEMA",
    "DIGEST_PAGE_SCHEMA",
    "ROOT_SCHEMA",
    "PREFLIGHT_SCHEMA",
    "DIGEST_SECTIONS",
    "SUPPORT_STATE_PRESENT",
    "SUPPORT_STATE_REFUSED",
    "SUPPORT_REFUSAL_REASONS",
    "CAPTURE_REFUSAL_REASONS",
    "PREFLIGHT_OK",
    "PREFLIGHT_REFUSED",
    "PREFLIGHT_CAP_CONFIGURATION_INVALID",
    "PREFLIGHT_SYNTHETIC_SIZE_OVER_CAP",
    "PREFLIGHT_AGGREGATE_UNBOUNDED",
    "INDEX_PAGE_MAX_BYTES",
    "DIGEST_PAGE_MAX_BYTES",
    "SIDECAR_PAGE_MAX_BYTES",
    "SIDECAR_ENTRY_MAX_RAW_BYTES",
    "SIDECAR_ENTRY_MAX_BYTES",
    "SIDECAR_AGGREGATE_MAX_BYTES",
    "ROOT_MAX_BYTES",
    "UNIVERSE_MAX_ROWS",
    "INDEX_ROW_MAX_FRAME_ID_BYTES",
    "INDEX_ROW_MAX_ABS_INT",
    "INGRESS_MAX_RECORDS",
    "INGRESS_MAX_FRAME_ID_BYTES",
    "INGRESS_MAX_SOURCE_LABEL_BYTES",
    "INGRESS_MAX_ABS_INT",
    "INGRESS_MAX_SHAPE_DIM",
    "PAYLOAD_MAX_SHAPE_DIM",
    "PAYLOAD_MAX_ABS_ELEMENT",
    "REFUSAL_FIELD_MAX_BYTES",
    "CAP_PLAN_DIGEST_DOMAIN",
    "CAP_PLAN_MAX_ABS_INT",
    "PLANLESS_REFUSAL_ROOT_MAX_BYTES",
    "PLANLESS_REFUSAL_REASON",
    "PLANLESS_REFUSAL_FIELD",
    "CAP_PLAN_FIELDS",
    "ROUTE_B_DIAGNOSTIC_EXPECTED_ROW_COUNT",
    "ROUTE_B_DIAGNOSTIC_CAP_PLAN",
    "CompleteSupportCapPlanV1",
    "CandidateSupportIndexRowV2",
    "CandidateSupportIndexPageV2",
    "MatcherSupportSidecarEntryV2",
    "MatcherSupportSidecarPageV2",
    "CandidateSupportDigestPageV2",
    "CandidateSupportIndexPageBindingV2",
    "CandidateSupportSidecarPageBindingV2",
    "CandidateSupportDigestPageBindingV2",
    "CandidateSupportDigestSectionV2",
    "CandidateSupportCaptureRootV2",
    "CompleteSupportCaptureV2",
    "CompleteSupportPreflightV1",
    "complete_support_universe_digest",
    "refused_complete_support",
    "validate_complete_support_capture",
    "build_complete_support_capture",
    "collect_complete_candidate_support",
    "collect_route_b_diagnostic_support",
    "preflight_complete_support",
]

# --- Schemas and domains ------------------------------------------------------

COMPLETE_SUPPORT_SCHEMA = "mc-photogrammetry-candidate-support-provenance-v2"
COMPLETE_SUPPORT_BASIS = "derived_from_complete_pre_filter_admitted_universe"
COMPLETE_SUPPORT_DIGEST_DOMAIN = "candidate_support_provenance_v2"
COMPLETE_SUPPORT_UNIVERSE_DOMAIN = "complete_support_universe"
COMPLETE_SUPPORT_UNIVERSE_SCHEMA = "mc-photogrammetry-complete-support-universe-v1"
COMPLETE_SUPPORT_CAPTURE_SCHEMA = "mc-photogrammetry-complete-support-capture-v2"
INDEX_PAGE_SCHEMA = "mc-photogrammetry-candidate-support-index-page-v2"
SIDECAR_ENTRY_SCHEMA = "mc-photogrammetry-matcher-support-sidecar-entry-v2"
SIDECAR_PAGE_SCHEMA = "mc-photogrammetry-matcher-support-sidecar-page-v2"
DIGEST_PAGE_SCHEMA = "mc-photogrammetry-candidate-support-digest-page-v2"
ROOT_SCHEMA = "mc-photogrammetry-candidate-support-capture-root-v2"
PREFLIGHT_SCHEMA = "mc-photogrammetry-complete-support-preflight-v1"

INDEX_PAGE_DIGEST_DOMAIN = "candidate_support_index_page_v2"
SIDECAR_PAGE_DIGEST_DOMAIN = "matcher_support_sidecar_page_v2"
DIGEST_PAGE_DIGEST_DOMAIN = "candidate_support_digest_page_v2"
DIGEST_SECTION_DIGEST_DOMAIN = "candidate_support_digest_section_v2"
CAP_PLAN_DIGEST_DOMAIN = "candidate_support_cap_plan_v2"

# Six parallel ordered sections, in the design's declared order.
DIGEST_SECTIONS = (
    "match_sha256",
    "envelope_sha256",
    "left_signature_sha256",
    "right_signature_sha256",
    "matcher_witness_sha256",
    "input_match_identity_sha256",
)

SUPPORT_STATE_PRESENT = "present"
SUPPORT_STATE_REFUSED = "refused"

# The bounded row-local support-outcome enum (design section 5). No free text.
SUPPORT_REFUSAL_REASONS = (
    "matcher_witness_unavailable",
    "support_payload_unavailable",
    "support_payload_over_cap",
    "source_tap_witness_unavailable",
    "source_tap_witness_shape_mismatch",
    "witness_digest_mismatch",
    "payload_digest_mismatch",
)

# The bounded capture-global refusal enum (design section 6.3/6.4).
CAPTURE_REFUSAL_REASONS = (
    "universe_count_mismatch",
    "universe_order_mismatch",
    "universe_identity_mismatch",
    "row_cap_exceeded",
    "index_row_over_cap",
    "ingress_over_cap",
    "index_page_over_cap",
    "digest_page_over_cap",
    "sidecar_page_over_cap",
    "sidecar_aggregate_over_cap",
    "root_over_cap",
    "cap_configuration_invalid",
)

PREFLIGHT_OK = "ok"
PREFLIGHT_REFUSED = "refused"
PREFLIGHT_CAP_CONFIGURATION_INVALID = "cap_configuration_invalid"
PREFLIGHT_SYNTHETIC_SIZE_OVER_CAP = "synthetic_size_over_cap"
PREFLIGHT_AGGREGATE_UNBOUNDED = "aggregate_unbounded"

# Declared caps (design section 6.1). ``SIDECAR_ENTRY_MAX_BYTES`` and
# ``SIDECAR_AGGREGATE_MAX_BYTES`` are placeholders the preflight must measure
# and the lead must set; the shipped values make the preflight refuse and
# propose measured replacements. Nothing here is an approved cap.
INDEX_PAGE_MAX_BYTES = 262_144
DIGEST_PAGE_MAX_BYTES = 262_144
SIDECAR_PAGE_MAX_BYTES = 262_144
SIDECAR_ENTRY_MAX_RAW_BYTES = 131_072
SIDECAR_ENTRY_MAX_BYTES = 131_072
SIDECAR_AGGREGATE_MAX_BYTES = 8_388_608
ROOT_MAX_BYTES = 262_144
UNIVERSE_MAX_ROWS = 4_096

# Declared serialized-field bounds. Every variable-width field this module
# serializes is bounded by exactly one of these, and the synthetic preflight
# measures its worst case at those bounds. They are declared contract limits,
# not approved caps. ``CandidateSupportIndexRowV2``,
# ``MatcherSupportSidecarEntryV2`` and ``CandidateSupportCaptureRootV2`` enforce
# them at construction, so a runtime-legal record can never encode beyond the
# preflight's measured ceiling.
INDEX_ROW_MAX_FRAME_ID_BYTES = 1_024
INDEX_ROW_MAX_ABS_INT = 2 ** 63 - 1
INGRESS_MAX_RECORDS = 16
INGRESS_MAX_FRAME_ID_BYTES = 1_024
INGRESS_MAX_SOURCE_LABEL_BYTES = 1_024
INGRESS_MAX_ABS_INT = 2 ** 63 - 1
INGRESS_MAX_SHAPE_DIM = 2 ** 31 - 1
# Array elements are declared-bounded so the sidecar canonical ceiling is finite
# even though ``source_shape``/``target_shape`` are serialized outside the raw
# array-byte accounting: shapes and every element live in ``[-PAYLOAD_MAX_ABS_ELEMENT,
# PAYLOAD_MAX_ABS_ELEMENT]``, whose longest canonical value is 11 characters
# (``-2147483647``).
PAYLOAD_MAX_SHAPE_DIM = 2 ** 31 - 1
PAYLOAD_MAX_ABS_ELEMENT = 2 ** 31 - 1
# The bounded capture-global refusal ``field`` code. Every internal refusal
# field is one of this module's own short field codes (``row.<i>.<name>``,
# ``ingress.<i>.<name>``, ``digest_page.<section>.<i>``, a plain field name or an
# exception class name); the bound makes a directly constructed refused root
# incapable of carrying an unbounded serialized string.
REFUSAL_FIELD_MAX_BYTES = 256
# Declared bound on every cap-plan integer. It makes the plan's canonical width
# finite, so the maximal planned-refusal root has a measurable ceiling and a
# directly constructed plan can never carry an unbounded decimal value.
CAP_PLAN_MAX_ABS_INT = 2 ** 63 - 1
# The fixed byte bound of a planless refusal root (one whose cap plan could not be
# resolved at all). It is deliberately smaller than ``ROOT_MAX_BYTES`` because such
# a root carries only the bounded reason/field pair and no plan.
PLANLESS_REFUSAL_ROOT_MAX_BYTES = 2_048
# A planless refusal exists for exactly one cause: the cap plan could not be
# resolved. Such a root must name that cause with this fixed reason/field pair, so
# no caller can mint an unattributable planless refusal (for example a planless
# ``row_cap_exceeded``) whose producing plan is unknown.
PLANLESS_REFUSAL_REASON = "cap_configuration_invalid"
PLANLESS_REFUSAL_FIELD = "cap_plan"
# The exact serialized field set of ``CompleteSupportCapPlanV1``. Plan validation
# requires the canonical record to carry exactly these keys, so a missing or extra
# serialized plan field is rejected rather than silently hashed.
CAP_PLAN_FIELDS = (
    "index_page_max_bytes", "digest_page_max_bytes", "sidecar_page_max_bytes",
    "sidecar_entry_max_raw_bytes", "sidecar_entry_max_bytes",
    "sidecar_aggregate_max_bytes", "root_max_bytes", "universe_max_rows")

_SYNTHETIC_DIGEST = "f" * 64
# Longest canonical finite float64 form (``-1.7976931348623157e+308``).
_SYNTHETIC_MAX_FLOAT = -1.7976931348623157e308


# --- Canonical encoding helpers ----------------------------------------------

def _canonical_bytes(payload: object) -> bytes:
    import json

    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _digest(domain: str, payload: object) -> str:
    return _sha({"domain": domain, "payload": payload})


def _ceil_to(value: int, unit: int) -> int:
    return ((value + unit - 1) // unit) * unit


def _serialized_string_bytes(value: str) -> int:
    """Canonical JSON width of one string, excluding its outer quotes.

    ``_canonical_bytes`` uses ``ensure_ascii=True``, so a non-ASCII character can
    encode to a ``\\uXXXX`` escape; measuring the canonical form is the only
    exact way to bound the serialized field.
    """
    return len(_canonical_bytes(value)) - 2


def _int_within_bound(value: object, bound: int) -> bool:
    return type(value) is int and -bound <= value <= bound


def _is_finite_float(value: object) -> bool:
    return type(value) is float and math.isfinite(value)


def _payload_over_declared_bounds(payload: "MatcherSupportPayloadV1") -> str:
    """Return the over-bound field of one support payload, or ``""``.

    Bounds the serialized ``source_shape``/``target_shape`` integers (which the
    raw array-byte gate does not count) and every array element.
    """
    for dim in payload.source_shape + payload.target_shape:
        if type(dim) is not int or dim <= 0 or dim > PAYLOAD_MAX_SHAPE_DIM:
            return "shape"
    for name, array in (("logical_pairs", payload.logical_pairs),
                        ("warp_source_rows", payload.warp_source_rows),
                        ("warp_source_columns", payload.warp_source_columns)):
        if array.size and (int(array.min()) < -PAYLOAD_MAX_ABS_ELEMENT
                           or int(array.max()) > PAYLOAD_MAX_ABS_ELEMENT):
            return name
    return ""


def _is_dimension_pair(values: object) -> bool:
    """Exactly two positive serialized size dimensions within the declared cap.

    ``FrameTransform`` is a plain dataclass, so ``source_size``/``crop_size``/
    ``output_size`` are not arity- or range-checked where they are declared. This
    module therefore refuses any tuple that is not exactly two positive integers
    bounded by ``INGRESS_MAX_SHAPE_DIM`` (the same bound the synthetic ingress
    record is measured at), so a runtime-legal pin can never serialize a longer
    or wider size list than the preflight measured.
    """
    if type(values) not in (tuple, list) or len(values) != 2:
        return False
    return all(type(item) is int and 0 < item <= INGRESS_MAX_SHAPE_DIM
               for item in values)


def _is_signed_pair(values: object) -> bool:
    """Exactly two signed coordinates within ``INGRESS_MAX_ABS_INT``."""
    if type(values) not in (tuple, list) or len(values) != 2:
        return False
    return all(_int_within_bound(item, INGRESS_MAX_ABS_INT) for item in values)


def _decoded_shape_over_cap(shape: object) -> bool:
    """Mirror :func:`frame_ingress._shape`'s arity/positivity contract.

    A construction-legal pin carries two or three positive dimensions and, when
    three, the third is the fixed RGB channel count. This check makes the
    declared ingress bound independent of the V1 constructor, so a tampered pin
    cannot smuggle a longer or wider decoded shape past the preflight.
    """
    if type(shape) not in (tuple, list) or len(shape) not in (2, 3):
        return True
    if any(type(item) is not int or item <= 0 or item > INGRESS_MAX_SHAPE_DIM
           for item in shape):
        return True
    return len(shape) == 3 and shape[2] != 3


def _ingress_record_over_cap(record: "FrameIngressProvenanceV1") -> str:
    """Return the over-bound field of one ingress pin, or ``""``."""
    if _serialized_string_bytes(record.frame_id) > INGRESS_MAX_FRAME_ID_BYTES:
        return "frame_id"
    if _serialized_string_bytes(record.source_label) > INGRESS_MAX_SOURCE_LABEL_BYTES:
        return "source_label"
    if record.byte_length > INGRESS_MAX_ABS_INT:
        return "byte_length"
    if _decoded_shape_over_cap(record.decoded_shape):
        return "decoded_shape"
    transform = record.transform
    if not _is_dimension_pair(transform.source_size):
        return "source_size"
    if not _is_signed_pair(transform.crop_origin):
        return "crop_origin"
    if transform.crop_size is not None and not _is_dimension_pair(
            transform.crop_size):
        return "crop_size"
    if transform.output_size is not None and not _is_dimension_pair(
            transform.output_size):
        return "output_size"
    return ""


def _bounded_refusal_field(field: object) -> str:
    """Return one deterministic, declared-bounded refusal field code.

    Internal refusals must stay atomic even when a caller-supplied plan makes a
    generated field code (for example a huge admitted index) exceed the declared
    bound, so an over-long string is truncated to fit rather than raising out of
    :func:`refused_complete_support`.
    """
    if type(field) is not str or not field:
        return "record"
    if _serialized_string_bytes(field) <= REFUSAL_FIELD_MAX_BYTES:
        return field
    trimmed = field[:REFUSAL_FIELD_MAX_BYTES]
    while trimmed and _serialized_string_bytes(trimmed) > REFUSAL_FIELD_MAX_BYTES:
        trimmed = trimmed[:-1]
    return trimmed or "record"


# The fixed operation code used when a caught raw exception carries no usable
# innermost frame. It is a source-declared identifier, never exception text.
_RAW_FAILURE_UNKNOWN_MODULE = "unknown"
_RAW_FAILURE_UNKNOWN_OPERATION = "unknown"
# Declared character bounds on the module and operation components of a raw
# failure field. They keep the composite localization code short before the
# outer byte bound (`REFUSAL_FIELD_MAX_BYTES`) is applied.
_RAW_FAILURE_MODULE_MAX_CHARS = 64
_RAW_FAILURE_OPERATION_MAX_CHARS = 64


def _raw_failure_token(value: object, *, allow_dot: bool,
                       limit: int, fallback: str) -> str:
    """Sanitize one static identifier to a bounded ASCII token.

    It is used only for interpreter-supplied code metadata (a module ``__name__``
    or a ``co_name``); it never receives exception text, payload or an object
    repr, so the result cannot carry data.
    """
    if type(value) is not str or not value:
        return fallback
    extra = "_." if allow_dot else "_"
    bounded = "".join(
        character if (character.isascii()
                      and (character.isalnum() or character in extra))
        else "_" for character in value)
    return bounded[:limit] or fallback


# The fixed canonical exception categories, in the declared match order of the
# broad raw-exception arms below. A caught exception is classified by
# ``isinstance`` against exactly these explicit caught bases, so a custom
# subclass can only ever surface as one of these built-in names; its own
# ``__name__`` (which a caller may set to an arbitrary, path-like string) is
# never read. No base here is a subclass of another, so this order is already
# deterministic and no category can shadow a more specific one.
_RAW_FAILURE_EXCEPTION_CATEGORIES = (
    AttributeError, TypeError, ValueError, IndexError, KeyError,
    OverflowError, ZeroDivisionError)
_RAW_FAILURE_UNKNOWN_EXCEPTION = "unknown"


def _raw_failure_exception_category(error: object) -> str:
    """Return one fixed canonical built-in category for a caught raw exception.

    Only ``isinstance`` against :data:`_RAW_FAILURE_EXCEPTION_CATEGORIES` is
    consulted. ``type(error).__name__`` is deliberately never read, because a
    dynamic subclass name is caller-controlled text and could carry a path or
    other data into the persisted refusal field. The category is therefore a
    source-declared identifier, never exception text or an object repr.
    """
    for category in _RAW_FAILURE_EXCEPTION_CATEGORIES:
        if isinstance(error, category):
            return category.__name__
    return _RAW_FAILURE_UNKNOWN_EXCEPTION


class _CollectorFailureSite:
    """The one mutable, data-free collector fact a traceback cannot supply.

    A raw exception's static traceback metadata names the innermost module,
    operation and source line that raised it, but it cannot name the collector's
    currently admitted row. This holder carries only that exact non-negative
    integer position; every other value clears it, so a non-row phase (assembly
    or validation) can never be mislabeled with a stale admitted row.
    """

    __slots__ = ("position",)

    def __init__(self) -> None:
        self.position = None

    def enter_row(self, position: object) -> None:
        self.position = (position if type(position) is int and position >= 0
                         else None)

    def leave_row(self) -> None:
        self.position = None


def _raw_failure_site_metadata(error: object) -> tuple[str, str, "int | None"]:
    """Innermost module, operation and source line from static code metadata.

    Only three data-free interpreter values of the innermost frame are read: the
    frame module's ``__name__`` (a static import identity such as
    ``mc_photogrammetry.reconstruction.plane_graph``), ``co_name`` (a source
    identifier) and ``tb_lineno`` (an exact positive ``int``). The file name,
    the exception text, the rendered traceback, frame locals, all other globals
    and every object repr are never read, so no path, payload, image-derived
    value or exception message can enter the refusal field.
    """
    traceback = getattr(error, "__traceback__", None)
    innermost = None
    while traceback is not None:
        innermost = traceback
        traceback = traceback.tb_next
    module = _RAW_FAILURE_UNKNOWN_MODULE
    operation = _RAW_FAILURE_UNKNOWN_OPERATION
    line: "int | None" = None
    if innermost is not None:
        frame = getattr(innermost, "tb_frame", None)
        namespace = getattr(frame, "f_globals", None)
        if type(namespace) is dict:
            module = _raw_failure_token(
                namespace.get("__name__"), allow_dot=True,
                limit=_RAW_FAILURE_MODULE_MAX_CHARS,
                fallback=_RAW_FAILURE_UNKNOWN_MODULE)
        code = getattr(frame, "f_code", None)
        operation = _raw_failure_token(
            getattr(code, "co_name", None), allow_dot=False,
            limit=_RAW_FAILURE_OPERATION_MAX_CHARS,
            fallback=_RAW_FAILURE_UNKNOWN_OPERATION)
        raised_at = getattr(innermost, "tb_lineno", None)
        if type(raised_at) is int and raised_at > 0:
            line = raised_at
    return module, operation, line


def _raw_failure_field(error: object, position: object = None) -> str:
    """Bounded deterministic localization code for one caught raw exception.

    It names the fixed canonical built-in exception category (never a custom
    subclass name), the innermost module, operation and source line from static
    code metadata, and the admitted row position only when that position is an
    exact non-negative ``int``. It is deterministic for a fixed source and can
    never carry ``str(error)``, a custom exception class name, rendered
    traceback text, a file path, frame locals, payload bytes or an object repr;
    the result always fits the declared ``REFUSAL_FIELD_MAX_BYTES`` bound.
    """
    module, operation, line = _raw_failure_site_metadata(error)
    code = f"{_raw_failure_exception_category(error)}:{module}.{operation}"
    if line is not None:
        code = f"{code}:{line}"
    if type(position) is int and position >= 0:
        code = f"{code}@row{position}"
    return _bounded_refusal_field(code)


def _row_field_over_cap(row: "CandidateSupportIndexRowV2") -> str:
    """Return the over-bound field of one index row, or ``""``."""
    if _serialized_string_bytes(row.left) > INDEX_ROW_MAX_FRAME_ID_BYTES:
        return "left"
    if _serialized_string_bytes(row.right) > INDEX_ROW_MAX_FRAME_ID_BYTES:
        return "right"
    for name in ("admitted_row_index", "group_index", "left_axis", "right_axis",
                 "quadrant_of_right", "left_plane", "right_plane",
                 "left_alternative_index", "right_alternative_index",
                 "overlap_cells"):
        if not _int_within_bound(getattr(row, name), INDEX_ROW_MAX_ABS_INT):
            return name
    if (row.match_index is not None
            and not _int_within_bound(row.match_index, INDEX_ROW_MAX_ABS_INT)):
        return "match_index"
    if row.cell_shift_rc is not None:
        for value in row.cell_shift_rc:
            if not _int_within_bound(value, INDEX_ROW_MAX_ABS_INT):
                return "cell_shift_rc"
    for name in ("left_b", "right_b", "delta", "tolerance"):
        if not _is_finite_float(getattr(row, name)):
            return name
    return ""


def _row_identity(row: "CandidateSupportIndexRowV2") -> list:
    return [row.left, row.left_plane, row.left_axis, row.right, row.right_plane,
            row.right_axis, row.quadrant_of_right, float(row.left_b),
            float(row.right_b)]


def complete_support_universe_digest(
        rows: "list[CandidateSupportIndexRowV2] | tuple[CandidateSupportIndexRowV2, ...]") -> str:
    """Ordered ``complete_support_universe`` digest over all admitted rows."""
    payload = [[row.admitted_row_index, _row_identity(row)] for row in rows]
    return _sha({"domain": COMPLETE_SUPPORT_UNIVERSE_DOMAIN,
                 "schema": COMPLETE_SUPPORT_UNIVERSE_SCHEMA,
                 "payload": payload})


# --- Records ------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CandidateSupportIndexRowV2:
    """One admitted candidate, kept or dropped, compact and array-free.

    The six per-row digest sequences live in the bounded digest pages, not here.
    ``support_payload_sha256`` is the one support-binding digest the index keeps:
    it lets the strict validator compare a sidecar entry to its own index row
    (design section 4.3) without re-inlining the payload.
    """

    admitted_row_index: int
    group_index: int
    left: str
    right: str
    left_axis: int
    right_axis: int
    quadrant_of_right: int
    left_plane: int
    right_plane: int
    left_alternative_index: int
    right_alternative_index: int
    left_b: float
    right_b: float
    delta: float
    tolerance: float
    disposition: str
    match_index: int | None
    overlap_cells: int
    cell_shift_rc: tuple[int, int] | None
    support_state: str
    support_refusal_reason: str | None
    support_payload_sha256: str | None
    orientation: str = CANDIDATE_SUPPORT_ORIENTATION

    def __post_init__(self) -> None:
        if (type(self.admitted_row_index) is not int or self.admitted_row_index < 0
                or type(self.group_index) is not int or self.group_index < 0
                or type(self.left) is not str or not self.left
                or type(self.right) is not str or not self.right
                or type(self.left_axis) is not int or type(self.right_axis) is not int
                or type(self.quadrant_of_right) is not int
                or type(self.left_plane) is not int or type(self.right_plane) is not int
                or type(self.left_alternative_index) is not int
                or self.left_alternative_index < 0
                or type(self.right_alternative_index) is not int
                or self.right_alternative_index < 0
                or self.disposition not in (plane_graph.SAME_AXIS_DISPOSITION_KEPT,
                                            plane_graph.SAME_AXIS_DISPOSITION_DROPPED)
                or type(self.overlap_cells) is not int or self.overlap_cells < 0
                or self.orientation != CANDIDATE_SUPPORT_ORIENTATION
                or self.support_state not in (SUPPORT_STATE_PRESENT,
                                              SUPPORT_STATE_REFUSED)):
            raise ValueError("candidate support V2 index row fields are malformed")
        if self.disposition == plane_graph.SAME_AXIS_DISPOSITION_KEPT:
            if type(self.match_index) is not int or self.match_index < 0:
                raise ValueError("kept V2 index row needs a match index")
        elif self.match_index is not None:
            raise ValueError("dropped V2 index row must not claim a match index")
        if self.cell_shift_rc is not None and (
                type(self.cell_shift_rc) is not tuple or len(self.cell_shift_rc) != 2
                or any(type(value) is not int for value in self.cell_shift_rc)):
            raise ValueError("V2 index row cell shift is malformed")
        if self.support_state == SUPPORT_STATE_PRESENT:
            if (self.support_refusal_reason is not None
                    or not _hex64(self.support_payload_sha256)):
                raise ValueError("present V2 index row needs its payload digest only")
        elif (self.support_refusal_reason not in SUPPORT_REFUSAL_REASONS
              or self.support_payload_sha256 is not None):
            raise ValueError("refused V2 index row needs exactly one bounded code")
        over = _row_field_over_cap(self)
        if over:
            raise CandidateSupportError("index_row_over_cap", over)

    def to_record(self) -> dict:
        return {
            "admitted_row_index": self.admitted_row_index,
            "cell_shift_rc": (None if self.cell_shift_rc is None
                              else list(self.cell_shift_rc)),
            "delta": float(self.delta),
            "disposition": self.disposition,
            "group_index": self.group_index,
            "left": self.left, "left_alternative_index": self.left_alternative_index,
            "left_axis": self.left_axis, "left_b": float(self.left_b),
            "left_plane": self.left_plane,
            "match_index": self.match_index,
            "orientation": self.orientation,
            "overlap_cells": self.overlap_cells,
            "quadrant_of_right": self.quadrant_of_right,
            "right": self.right, "right_alternative_index": self.right_alternative_index,
            "right_axis": self.right_axis, "right_b": float(self.right_b),
            "right_plane": self.right_plane,
            "support_payload_sha256": self.support_payload_sha256,
            "support_refusal_reason": self.support_refusal_reason,
            "support_state": self.support_state,
            "tolerance": float(self.tolerance),
        }


def _index_page_payload(page: "CandidateSupportIndexPageV2", *, include_digest: bool) -> dict:
    payload = {
        "first_admitted_row_index": page.first_admitted_row_index,
        "page_index": page.page_index,
        "row_count": len(page.rows),
        "rows": [row.to_record() for row in page.rows],
        "schema": page.schema,
    }
    if include_digest:
        payload["page_sha256"] = page.page_sha256
    return payload


@dataclass(frozen=True, slots=True)
class CandidateSupportIndexPageV2:
    """A consecutive, bounded slice of the single canonical admitted order."""

    page_index: int
    first_admitted_row_index: int
    rows: tuple[CandidateSupportIndexRowV2, ...]
    page_sha256: str = ""
    schema: str = INDEX_PAGE_SCHEMA

    def __post_init__(self) -> None:
        if (self.schema != INDEX_PAGE_SCHEMA
                or type(self.page_index) is not int or self.page_index < 0
                or type(self.first_admitted_row_index) is not int
                or self.first_admitted_row_index < 0
                or not isinstance(self.rows, tuple) or not self.rows
                or any(not isinstance(row, CandidateSupportIndexRowV2)
                       for row in self.rows)
                or self.rows[0].admitted_row_index != self.first_admitted_row_index):
            raise ValueError("candidate support V2 index page is malformed")
        for offset, row in enumerate(self.rows):
            if row.admitted_row_index != self.first_admitted_row_index + offset:
                raise ValueError("candidate support V2 index page is not consecutive")
        digest = _digest(INDEX_PAGE_DIGEST_DOMAIN,
                         _index_page_payload(self, include_digest=False))
        if self.page_sha256 not in ("", digest):
            raise ValueError("candidate support V2 index page digest mismatch")
        object.__setattr__(self, "page_sha256", digest)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def to_record(self) -> dict:
        return _index_page_payload(self, include_digest=True)


@dataclass(frozen=True, slots=True)
class MatcherSupportSidecarEntryV2:
    """One present row's retrievable array material, bound to its index row."""

    admitted_row_index: int
    payload: MatcherSupportPayloadV1
    payload_sha256: str = dataclass_field(init=False, default="")
    logical_pairs_sha256: str = dataclass_field(init=False, default="")
    warp_source_rows_sha256: str = dataclass_field(init=False, default="")
    warp_source_columns_sha256: str = dataclass_field(init=False, default="")
    source_shape: tuple[int, int] = dataclass_field(init=False, default=())
    target_shape: tuple[int, int] = dataclass_field(init=False, default=())
    overlap_cells: int = dataclass_field(init=False, default=-1)
    schema: str = SIDECAR_ENTRY_SCHEMA

    def __post_init__(self) -> None:
        if (self.schema != SIDECAR_ENTRY_SCHEMA
                or type(self.admitted_row_index) is not int
                or self.admitted_row_index < 0
                or not isinstance(self.payload, MatcherSupportPayloadV1)):
            raise ValueError("matcher support V2 sidecar entry is malformed")
        over = _payload_over_declared_bounds(self.payload)
        if over:
            raise CandidateSupportError(SUPPORT_PAYLOAD_OVER_CAP, over)
        object.__setattr__(self, "payload_sha256", self.payload.payload_sha256)
        object.__setattr__(self, "logical_pairs_sha256", hashlib.sha256(
            self.payload.logical_pairs.tobytes(order="C")).hexdigest())
        object.__setattr__(self, "warp_source_rows_sha256", hashlib.sha256(
            self.payload.warp_source_rows.tobytes(order="C")).hexdigest())
        object.__setattr__(self, "warp_source_columns_sha256", hashlib.sha256(
            self.payload.warp_source_columns.tobytes(order="C")).hexdigest())
        object.__setattr__(self, "source_shape", self.payload.source_shape)
        object.__setattr__(self, "target_shape", self.payload.target_shape)
        object.__setattr__(self, "overlap_cells", len(self.payload.logical_pairs))

    def to_record(self) -> dict:
        return {
            "admitted_row_index": self.admitted_row_index,
            "logical_pairs_sha256": self.logical_pairs_sha256,
            "overlap_cells": self.overlap_cells,
            "payload": self.payload.to_record(),
            "payload_sha256": self.payload_sha256,
            "schema": self.schema,
            "source_shape": list(self.source_shape),
            "target_shape": list(self.target_shape),
            "warp_source_columns_sha256": self.warp_source_columns_sha256,
            "warp_source_rows_sha256": self.warp_source_rows_sha256,
        }


def _sidecar_page_payload(page: "MatcherSupportSidecarPageV2", *,
                          include_digest: bool) -> dict:
    payload = {
        "entries": [entry.to_record() for entry in page.entries],
        "entry_count": len(page.entries),
        "first_admitted_row_index": page.first_admitted_row_index,
        "page_index": page.page_index,
        "schema": page.schema,
    }
    if include_digest:
        payload["page_sha256"] = page.page_sha256
    return payload


@dataclass(frozen=True, slots=True)
class MatcherSupportSidecarPageV2:
    """A bounded slice of present-row sidecar entries; one entry is atomic."""

    page_index: int
    first_admitted_row_index: int
    entries: tuple[MatcherSupportSidecarEntryV2, ...]
    entry_count: int = dataclass_field(init=False, default=0)
    page_sha256: str = ""
    schema: str = SIDECAR_PAGE_SCHEMA

    def __post_init__(self) -> None:
        if (self.schema != SIDECAR_PAGE_SCHEMA
                or type(self.page_index) is not int or self.page_index < 0
                or type(self.first_admitted_row_index) is not int
                or self.first_admitted_row_index < 0
                or not isinstance(self.entries, tuple) or not self.entries
                or any(not isinstance(entry, MatcherSupportSidecarEntryV2)
                       for entry in self.entries)
                or self.entries[0].admitted_row_index != self.first_admitted_row_index):
            raise ValueError("matcher support V2 sidecar page is malformed")
        previous = self.entries[0].admitted_row_index
        for entry in self.entries[1:]:
            if entry.admitted_row_index <= previous:
                raise ValueError("matcher support V2 sidecar entries are not ordered")
            previous = entry.admitted_row_index
        object.__setattr__(self, "entry_count", len(self.entries))
        digest = _digest(SIDECAR_PAGE_DIGEST_DOMAIN,
                         _sidecar_page_payload(self, include_digest=False))
        if self.page_sha256 not in ("", digest):
            raise ValueError("matcher support V2 sidecar page digest mismatch")
        object.__setattr__(self, "page_sha256", digest)

    def to_record(self) -> dict:
        return _sidecar_page_payload(self, include_digest=True)


def _digest_page_payload(page: "CandidateSupportDigestPageV2", *,
                         include_digest: bool) -> dict:
    payload = {
        "digest_count": len(page.digests),
        "digests": list(page.digests),
        "first_admitted_row_index": page.first_admitted_row_index,
        "page_index": page.page_index,
        "schema": page.schema,
        "section": page.section,
    }
    if include_digest:
        payload["page_sha256"] = page.page_sha256
    return payload


@dataclass(frozen=True, slots=True)
class CandidateSupportDigestPageV2:
    """A bounded run of one digest section; ``None`` keeps its ordered slot."""

    section: str
    page_index: int
    first_admitted_row_index: int
    digests: tuple[str | None, ...]
    digest_count: int = dataclass_field(init=False, default=0)
    page_sha256: str = ""
    schema: str = DIGEST_PAGE_SCHEMA

    def __post_init__(self) -> None:
        if (self.schema != DIGEST_PAGE_SCHEMA
                or self.section not in DIGEST_SECTIONS
                or type(self.page_index) is not int or self.page_index < 0
                or type(self.first_admitted_row_index) is not int
                or self.first_admitted_row_index < 0
                or not isinstance(self.digests, tuple) or not self.digests
                or any(value is not None and not _hex64(value)
                       for value in self.digests)):
            raise ValueError("candidate support V2 digest page is malformed")
        object.__setattr__(self, "digest_count", len(self.digests))
        digest = _digest(DIGEST_PAGE_DIGEST_DOMAIN,
                         _digest_page_payload(self, include_digest=False))
        if self.page_sha256 not in ("", digest):
            raise ValueError("candidate support V2 digest page digest mismatch")
        object.__setattr__(self, "page_sha256", digest)

    def to_record(self) -> dict:
        return _digest_page_payload(self, include_digest=True)


@dataclass(frozen=True, slots=True)
class CandidateSupportIndexPageBindingV2:
    page_index: int
    sha256: str
    row_count: int

    def __post_init__(self) -> None:
        if (type(self.page_index) is not int or self.page_index < 0
                or not _hex64(self.sha256)
                or type(self.row_count) is not int or self.row_count <= 0):
            raise ValueError("candidate support V2 index binding is malformed")

    def to_record(self) -> dict:
        return {"page_index": self.page_index, "row_count": self.row_count,
                "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class CandidateSupportSidecarPageBindingV2:
    page_index: int
    sha256: str
    entry_count: int

    def __post_init__(self) -> None:
        if (type(self.page_index) is not int or self.page_index < 0
                or not _hex64(self.sha256)
                or type(self.entry_count) is not int or self.entry_count <= 0):
            raise ValueError("candidate support V2 sidecar binding is malformed")

    def to_record(self) -> dict:
        return {"entry_count": self.entry_count, "page_index": self.page_index,
                "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class CandidateSupportDigestPageBindingV2:
    page_index: int
    first_admitted_row_index: int
    digest_count: int
    sha256: str

    def __post_init__(self) -> None:
        if (type(self.page_index) is not int or self.page_index < 0
                or type(self.first_admitted_row_index) is not int
                or self.first_admitted_row_index < 0
                or type(self.digest_count) is not int or self.digest_count <= 0
                or not _hex64(self.sha256)):
            raise ValueError("candidate support V2 digest page binding is malformed")

    def to_record(self) -> dict:
        return {"digest_count": self.digest_count,
                "first_admitted_row_index": self.first_admitted_row_index,
                "page_index": self.page_index, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class CandidateSupportDigestSectionV2:
    section: str
    page_count: int
    section_sha256: str
    pages: tuple[CandidateSupportDigestPageBindingV2, ...]

    def __post_init__(self) -> None:
        if (self.section not in DIGEST_SECTIONS
                or type(self.page_count) is not int or self.page_count != len(self.pages)
                or not _hex64(self.section_sha256)
                or not isinstance(self.pages, tuple)
                or any(not isinstance(page, CandidateSupportDigestPageBindingV2)
                       for page in self.pages)):
            raise ValueError("candidate support V2 digest section is malformed")

    def to_record(self) -> dict:
        return {"page_count": self.page_count, "pages": [page.to_record()
                                                         for page in self.pages],
                "section": self.section, "section_sha256": self.section_sha256}


def _root_payload(root: "CandidateSupportCaptureRootV2", *,
                  include_digest: bool) -> dict:
    if root.status == "refused":
        return {"cap_plan": (None if root.cap_plan is None
                             else root.cap_plan.as_record()),
                "cap_plan_sha256": root.cap_plan_sha256,
                "field": root.field, "reason": root.reason, "schema": root.schema,
                "status": "refused"}
    payload = {
        "basis": root.basis,
        "cap_plan": (None if root.cap_plan is None else root.cap_plan.as_record()),
        "cap_plan_sha256": root.cap_plan_sha256,
        "declared_universe_cap": root.declared_universe_cap,
        "digest_page_count": root.digest_page_count,
        "digest_sections": [section.to_record() for section in root.digest_sections],
        "failed_assignment_sha256": root.failed_assignment_sha256,
        "index_page_count": root.index_page_count,
        "index_pages": [binding.to_record() for binding in root.index_pages],
        "ingress": [record.to_record() for record in root.ingress],
        "input_match_identity_sha256": root.input_match_identity_sha256,
        "row_count": root.row_count,
        "schema": root.schema,
        "sidecar_page_count": root.sidecar_page_count,
        "sidecar_pages": [binding.to_record() for binding in root.sidecar_pages],
        "status": root.status,
        "whole_universe_identity_sha256": root.whole_universe_identity_sha256,
    }
    if include_digest:
        payload["record_sha256"] = root.record_sha256
    return payload


@dataclass(frozen=True, slots=True)
class CandidateSupportCaptureRootV2:
    """The small binding root; it never carries arrays or digest sequences."""

    status: str
    basis: str | None
    reason: str | None
    field: str | None
    failed_assignment_sha256: str
    whole_universe_identity_sha256: str
    input_match_identity_sha256: str
    row_count: int
    declared_universe_cap: int
    index_pages: tuple[CandidateSupportIndexPageBindingV2, ...]
    sidecar_pages: tuple[CandidateSupportSidecarPageBindingV2, ...]
    digest_sections: tuple[CandidateSupportDigestSectionV2, ...]
    ingress: tuple[FrameIngressProvenanceV1, ...]
    cap_plan: "CompleteSupportCapPlanV1 | None"
    schema: str = ROOT_SCHEMA
    cap_plan_sha256: str = ""
    record_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema != ROOT_SCHEMA or self.status not in ("complete", "refused"):
            raise ValueError("candidate support V2 root is malformed")
        if self.status == "refused":
            if (self.basis is not None or not self.reason
                    or self.reason not in CAPTURE_REFUSAL_REASONS
                    or type(self.field) is not str
                    or not self.field
                    or _serialized_string_bytes(self.field) > REFUSAL_FIELD_MAX_BYTES
                    or self.row_count != 0 or self.declared_universe_cap != 0
                    or self.failed_assignment_sha256
                    or self.whole_universe_identity_sha256
                    or self.input_match_identity_sha256
                    or self.index_pages or self.sidecar_pages
                    or self.digest_sections or self.ingress):
                raise ValueError("refused V2 root must be atomic and empty")
            if self.cap_plan is not None:
                # A refusal produced after a plan was resolved must name and hash
                # the plan that produced it instead of silently dropping it. The
                # plan's own eight values are re-validated here rather than trusted
                # from construction time, because ``object.__setattr__`` can still
                # mutate a frozen dataclass.
                if (not isinstance(self.cap_plan, CompleteSupportCapPlanV1)
                        or _cap_plan_field_over_cap(self.cap_plan)):
                    raise ValueError("refused V2 root cap plan is malformed")
                plan_digest = _digest(CAP_PLAN_DIGEST_DOMAIN,
                                      self.cap_plan.as_record())
                if self.cap_plan_sha256 not in ("", plan_digest):
                    raise ValueError("refused V2 cap plan digest mismatch")
                object.__setattr__(self, "cap_plan_sha256", plan_digest)
            else:
                # A planless refusal is legal for exactly one cause: plan
                # resolution itself failed. It must carry the fixed
                # configuration-invalid pair, so it cannot silently omit the plan
                # that produced an ordinary refusal.
                if (self.reason != PLANLESS_REFUSAL_REASON
                        or self.field != PLANLESS_REFUSAL_FIELD):
                    raise ValueError(
                        "planless refused V2 root must be the fixed "
                        "cap-configuration refusal")
                if self.cap_plan_sha256:
                    raise ValueError(
                        "planless refused V2 root must not claim a cap plan digest")
            refusal_bound = (ROOT_MAX_BYTES if self.cap_plan is not None
                             else PLANLESS_REFUSAL_ROOT_MAX_BYTES)
            if (len(_canonical_bytes(_root_payload(self, include_digest=False)))
                    > refusal_bound):
                raise ValueError(
                    "refused V2 root exceeds its declared refusal byte bound")
        else:
            if (self.basis != COMPLETE_SUPPORT_BASIS or self.reason is not None
                    or self.field is not None
                    or type(self.row_count) is not int or self.row_count <= 0
                    or type(self.declared_universe_cap) is not int
                    or self.declared_universe_cap < self.row_count
                    or not isinstance(self.cap_plan, CompleteSupportCapPlanV1)
                    or self.declared_universe_cap != self.cap_plan.universe_max_rows
                    or self.row_count > self.cap_plan.universe_max_rows
                    or not _hex64(self.failed_assignment_sha256)
                    or not _hex64(self.whole_universe_identity_sha256)
                    or not _hex64(self.input_match_identity_sha256)
                    or not all(isinstance(binding, CandidateSupportIndexPageBindingV2)
                               for binding in self.index_pages)
                    or not all(isinstance(binding, CandidateSupportSidecarPageBindingV2)
                               for binding in self.sidecar_pages)
                    or not all(isinstance(section, CandidateSupportDigestSectionV2)
                               for section in self.digest_sections)
                    or any(not isinstance(record, FrameIngressProvenanceV1)
                           for record in self.ingress)):
                raise ValueError("complete V2 root is malformed")
            if _cap_plan_field_over_cap(self.cap_plan):
                raise ValueError("complete V2 root cap plan fields are out of bounds")
            plan_digest = _digest(CAP_PLAN_DIGEST_DOMAIN, self.cap_plan.as_record())
            if self.cap_plan_sha256 not in ("", plan_digest):
                raise ValueError("candidate support V2 cap plan digest mismatch")
            object.__setattr__(self, "cap_plan_sha256", plan_digest)
        if len(self.ingress) > INGRESS_MAX_RECORDS:
            raise CandidateSupportError("ingress_over_cap", "ingress_count")
        for position, record in enumerate(self.ingress):
            over = _ingress_record_over_cap(record)
            if over:
                raise CandidateSupportError("ingress_over_cap",
                                            f"ingress.{position}.{over}")
        digest = _digest(COMPLETE_SUPPORT_DIGEST_DOMAIN,
                         _root_payload(self, include_digest=False))
        if self.record_sha256 not in ("", digest):
            raise ValueError("candidate support V2 root digest mismatch")
        object.__setattr__(self, "record_sha256", digest)

    @property
    def index_page_count(self) -> int:
        return len(self.index_pages)

    @property
    def sidecar_page_count(self) -> int:
        return len(self.sidecar_pages)

    @property
    def digest_page_count(self) -> int:
        return sum(section.page_count for section in self.digest_sections)

    def to_record(self) -> dict:
        return _root_payload(self, include_digest=True)


@dataclass(frozen=True, slots=True)
class CompleteSupportCaptureV2:
    """The staged capture: root plus every page, all-or-nothing."""

    root: CandidateSupportCaptureRootV2
    index_pages: tuple[CandidateSupportIndexPageV2, ...] = ()
    digest_pages: tuple[CandidateSupportDigestPageV2, ...] = ()
    sidecar_pages: tuple[MatcherSupportSidecarPageV2, ...] = ()
    schema: str = COMPLETE_SUPPORT_CAPTURE_SCHEMA

    def __post_init__(self) -> None:
        validate_complete_support_capture(self)

    @property
    def status(self) -> str:
        return self.root.status

    @property
    def row_count(self) -> int:
        return self.root.row_count

    def to_record(self) -> dict:
        return {
            "digest_pages": [page.to_record() for page in self.digest_pages],
            "index_pages": [page.to_record() for page in self.index_pages],
            "root": self.root.to_record(),
            "schema": self.schema,
            "sidecar_pages": [page.to_record() for page in self.sidecar_pages],
        }


def refused_complete_support(
        reason: str, field: str,
        plan: "CompleteSupportCapPlanV1 | None" = None) -> CompleteSupportCaptureV2:
    """One typed, atomic, zero-row capture refusal.

    When ``plan`` is a resolved :class:`CompleteSupportCapPlanV1` it is preserved
    and hashed inside the refused root so a refusal names the plan that produced
    it. When no plan could be resolved at all (``plan`` is ``None`` or not a
    value-valid plan) the refusal is planless and is forced to the fixed
    ``cap_configuration_invalid``/"cap_plan" pair: an ordinary refusal code must
    never be emitted without the plan that produced it. The coercion keeps an
    internal refusal atomic instead of raising out of this builder; the root
    constructor and the validator reject any planless root that is not that pair.
    """
    resolved_plan = (plan if isinstance(plan, CompleteSupportCapPlanV1)
                     and not _cap_plan_field_over_cap(plan) else None)
    if resolved_plan is None:
        resolved_reason = PLANLESS_REFUSAL_REASON
        resolved_field = PLANLESS_REFUSAL_FIELD
    else:
        resolved_reason = (reason if reason in CAPTURE_REFUSAL_REASONS
                           else "universe_identity_mismatch")
        resolved_field = _bounded_refusal_field(field)
    root = CandidateSupportCaptureRootV2(
        status="refused", basis=None, reason=resolved_reason,
        field=resolved_field,
        failed_assignment_sha256="", whole_universe_identity_sha256="",
        input_match_identity_sha256="", row_count=0, declared_universe_cap=0,
        index_pages=(), sidecar_pages=(), digest_sections=(), ingress=(),
        cap_plan=resolved_plan)
    return CompleteSupportCaptureV2(root=root)


def _digest_section_from_pages(
        section: str,
        pages: tuple[CandidateSupportDigestPageV2, ...]) -> CandidateSupportDigestSectionV2:
    if not pages:
        raise CandidateSupportError("universe_identity_mismatch", f"section.{section}")
    bindings = tuple(
        CandidateSupportDigestPageBindingV2(
            page.page_index, page.first_admitted_row_index, page.digest_count,
            page.page_sha256) for page in pages)
    section_sha = _digest(DIGEST_SECTION_DIGEST_DOMAIN, {
        "pages": [page.page_sha256 for page in pages], "section": section})
    return CandidateSupportDigestSectionV2(
        section=section, page_count=len(pages), section_sha256=section_sha,
        pages=bindings)


def _index_page_content_digest(page: "CandidateSupportIndexPageV2") -> str:
    """Recompute an index page digest from its current fields and rows."""
    return _digest(INDEX_PAGE_DIGEST_DOMAIN,
                   _index_page_payload(page, include_digest=False))


def _digest_page_content_digest(page: "CandidateSupportDigestPageV2") -> str:
    """Recompute a digest page digest from its current ordered values."""
    return _digest(DIGEST_PAGE_DIGEST_DOMAIN,
                   _digest_page_payload(page, include_digest=False))


def _sidecar_page_content_digest(page: "MatcherSupportSidecarPageV2") -> str:
    """Recompute a sidecar page digest from its current entries."""
    return _digest(SIDECAR_PAGE_DIGEST_DOMAIN,
                   _sidecar_page_payload(page, include_digest=False))


def _sidecar_entry_content_mismatch(
        entry: "MatcherSupportSidecarEntryV2") -> str:
    """Return the mismatched sidecar-entry field, or ``""`` when it replays.

    Every cached digest, shape and count is recomputed from the entry's **current**
    payload arrays and re-bound to the payload's own digest. A frozen dataclass or
    a stale cached hash is never treated as tamper protection, so a swapped
    same-shape payload array cannot ride a previously computed digest.
    """
    payload = entry.payload
    if not isinstance(payload, MatcherSupportPayloadV1):
        return "payload"
    if (type(payload.source_shape) is not tuple or len(payload.source_shape) != 2
            or type(payload.target_shape) is not tuple
            or len(payload.target_shape) != 2):
        return "shapes"
    logical = payload.logical_pairs
    rows = payload.warp_source_rows
    columns = payload.warp_source_columns
    if (not isinstance(logical, np.ndarray) or logical.ndim != 2
            or logical.shape[1] != 4 or logical.dtype != np.int64
            or logical.flags.writeable):
        return "logical_pairs"
    if (not isinstance(rows, np.ndarray) or not isinstance(columns, np.ndarray)
            or rows.dtype != np.int64 or columns.dtype != np.int64
            or rows.flags.writeable or columns.flags.writeable):
        return "warp_arrays"
    if (payload.source_shape != entry.source_shape
            or payload.target_shape != entry.target_shape
            or rows.shape != payload.target_shape
            or columns.shape != payload.target_shape):
        return "shapes"
    if entry.overlap_cells != len(logical):
        return "overlap_cells"
    try:
        over = _payload_over_declared_bounds(payload)
    except (AttributeError, TypeError, ValueError):
        return "declared_bounds"
    if over:
        return over
    if hashlib.sha256(logical.tobytes(order="C")).hexdigest() != (
            entry.logical_pairs_sha256):
        return "logical_pairs_sha256"
    if hashlib.sha256(rows.tobytes(order="C")).hexdigest() != (
            entry.warp_source_rows_sha256):
        return "warp_source_rows_sha256"
    if hashlib.sha256(columns.tobytes(order="C")).hexdigest() != (
            entry.warp_source_columns_sha256):
        return "warp_source_columns_sha256"
    current = payload.to_record()
    current.pop("payload_sha256", None)
    recomputed = _sha(current)
    if (recomputed != entry.payload_sha256
            or recomputed != payload.payload_sha256):
        return "payload_sha256"
    return ""


def validate_complete_support_capture(capture: object) -> None:
    """Strict, fail-closed validation of one staged V2 capture."""
    if not isinstance(capture, CompleteSupportCaptureV2):
        raise ValueError("complete support capture has an invalid type")
    if capture.schema != COMPLETE_SUPPORT_CAPTURE_SCHEMA:
        raise ValueError("complete support capture schema mismatch")
    root = capture.root
    if not isinstance(root, CandidateSupportCaptureRootV2):
        raise ValueError("complete support capture root is malformed")
    if root.record_sha256 != _digest(COMPLETE_SUPPORT_DIGEST_DOMAIN,
                                     _root_payload(root, include_digest=False)):
        raise ValueError("complete support root digest does not replay")
    if root.status == "refused":
        if capture.index_pages or capture.digest_pages or capture.sidecar_pages:
            raise ValueError("refused complete support capture persisted pages")
        # The reason code is re-checked here because ``object.__setattr__`` can
        # rewrite a frozen root after construction.
        if (type(root.reason) is not str
                or root.reason not in CAPTURE_REFUSAL_REASONS):
            raise ValueError("refused complete support reason is not a bounded code")
        if root.cap_plan is not None:
            # A planned refusal must name and hash the plan that produced it, and
            # every plan value is re-validated from the current object rather than
            # trusted from construction time.
            if (not isinstance(root.cap_plan, CompleteSupportCapPlanV1)
                    or _cap_plan_field_over_cap(root.cap_plan)):
                raise ValueError(
                    "refused complete support cap plan is malformed")
            if root.cap_plan_sha256 != _digest(
                    CAP_PLAN_DIGEST_DOMAIN, root.cap_plan.as_record()):
                raise ValueError(
                    "refused complete support cap plan digest does not replay")
            refusal_bound = ROOT_MAX_BYTES
        else:
            # A planless refusal is legal only for plan-resolution failure, so it
            # must be exactly the fixed configuration-invalid pair even when its
            # own digests are internally consistent.
            if (root.reason != PLANLESS_REFUSAL_REASON
                    or root.field != PLANLESS_REFUSAL_FIELD):
                raise ValueError(
                    "planless refused complete support is not the fixed "
                    "cap-configuration refusal")
            if root.cap_plan_sha256:
                raise ValueError(
                    "planless refused complete support claims a cap plan digest")
            refusal_bound = PLANLESS_REFUSAL_ROOT_MAX_BYTES
        if len(_canonical_bytes(root.to_record())) > refusal_bound:
            raise ValueError(
                "refused complete support root exceeds its declared byte bound")
        return
    plan = root.cap_plan
    if (not isinstance(plan, CompleteSupportCapPlanV1)
            or _cap_plan_field_over_cap(plan)):
        raise ValueError(
            "complete support root cap plan is missing or out of bounds")
    if root.cap_plan_sha256 != _digest(CAP_PLAN_DIGEST_DOMAIN, plan.as_record()):
        raise ValueError("complete support cap plan digest does not replay")
    if root.declared_universe_cap != plan.universe_max_rows:
        raise ValueError("complete support declared universe cap does not match the plan")
    if root.row_count > plan.universe_max_rows:
        raise ValueError("complete support row count exceeds the declared universe cap")
    if len(_canonical_bytes(root.to_record())) > plan.root_max_bytes:
        raise ValueError("complete support root exceeds the declared root cap")
    if len(capture.index_pages) != root.index_page_count:
        raise ValueError("complete support index page count does not replay")
    if len(capture.sidecar_pages) != root.sidecar_page_count:
        raise ValueError("complete support sidecar page count does not replay")
    if len(capture.digest_pages) != root.digest_page_count:
        raise ValueError("complete support digest page count does not replay")
    rows: list[CandidateSupportIndexRowV2] = []
    for position, (page, binding) in enumerate(zip(capture.index_pages,
                                                   root.index_pages)):
        if (not isinstance(page, CandidateSupportIndexPageV2)
                or not isinstance(page.rows, tuple)
                or any(not isinstance(row, CandidateSupportIndexRowV2)
                       for row in page.rows)
                or page.page_index != position or binding.page_index != position
                or page.page_sha256 != binding.sha256
                or page.page_sha256 != _index_page_content_digest(page)
                or page.row_count != binding.row_count):
            raise ValueError("complete support index page binding does not replay")
        expected_first = 0 if not rows else rows[-1].admitted_row_index + 1
        if page.first_admitted_row_index != expected_first:
            raise ValueError("complete support index pages are not consecutive")
        rows.extend(page.rows)
    if len(rows) != root.row_count:
        raise ValueError("complete support index row count does not replay")
    for position, row in enumerate(rows):
        if row.admitted_row_index != position:
            raise ValueError("complete support index rows are not the admitted order")
    entries: list[MatcherSupportSidecarEntryV2] = []
    present = [row for row in rows if row.support_state == SUPPORT_STATE_PRESENT]
    for position, (page, binding) in enumerate(zip(capture.sidecar_pages,
                                                   root.sidecar_pages)):
        if (not isinstance(page, MatcherSupportSidecarPageV2)
                or not isinstance(page.entries, tuple)
                or any(not isinstance(entry, MatcherSupportSidecarEntryV2)
                       for entry in page.entries)
                or page.page_index != position or binding.page_index != position
                or page.page_sha256 != binding.sha256
                or page.page_sha256 != _sidecar_page_content_digest(page)
                or page.entry_count != binding.entry_count
                or page.entry_count != len(page.entries)):
            raise ValueError("complete support sidecar page binding does not replay")
        # Present rows may be sparse, so sidecar pages are **not** consecutive in
        # admitted-row space: a page must begin at the admitted index of the
        # present row its first entry actually binds, and the flattened entries
        # must stay strictly increasing (no gap renumbering, reorder or
        # duplicate). The strict count/identity binding to the present rows is
        # enforced below; this only replays page sequencing.
        if entries and page.first_admitted_row_index <= entries[-1].admitted_row_index:
            raise ValueError(
                "complete support sidecar pages are not in admitted order")
        # Bound check before indexing: a well-formed sidecar has at most one entry
        # per present row, so once the present rows are exhausted an extra page
        # cannot bind and must fail closed as ValueError instead of leaking an
        # IndexError from present[len(entries)].
        if len(entries) >= len(present):
            raise ValueError(
                "complete support sidecar page has no present row to bind")
        if page.first_admitted_row_index != present[len(entries)].admitted_row_index:
            raise ValueError(
                "complete support sidecar page does not start at its first "
                "present row")
        entries.extend(page.entries)
    if len(entries) != len(present):
        raise ValueError("complete support sidecar does not cover the present rows")
    for row, entry in zip(present, entries):
        mismatch = _sidecar_entry_content_mismatch(entry)
        if mismatch:
            raise ValueError(
                "complete support sidecar entry does not replay: " + mismatch)
        if (row.admitted_row_index != entry.admitted_row_index
                or row.support_payload_sha256 != entry.payload_sha256):
            raise ValueError("complete support sidecar entry does not bind its index row")
    pages_by_section: dict[str, list[CandidateSupportDigestPageV2]] = {}
    for page in capture.digest_pages:
        if not isinstance(page, CandidateSupportDigestPageV2):
            raise ValueError("complete support digest page is malformed")
        pages_by_section.setdefault(page.section, []).append(page)
    if set(pages_by_section) != set(DIGEST_SECTIONS):
        raise ValueError("complete support digest sections are incomplete")
    for section in DIGEST_SECTIONS:
        pages = pages_by_section[section]
        expected_first = 0
        for position, page in enumerate(pages):
            if (not isinstance(page.digests, tuple)
                    or page.page_index != position
                    or page.first_admitted_row_index != expected_first
                    or page.digest_count != len(page.digests)
                    or page.page_sha256 != _digest_page_content_digest(page)):
                raise ValueError("complete support digest pages are not consecutive")
            expected_first += page.digest_count
        if expected_first != root.row_count:
            raise ValueError("complete support digest section count does not replay")
        recomputed = _digest_section_from_pages(section, tuple(pages))
        binding = next((item for item in root.digest_sections
                        if item.section == section), None)
        if binding is None or binding.to_record() != recomputed.to_record():
            raise ValueError("complete support digest section binding does not replay")
    if len(root.digest_sections) != len(DIGEST_SECTIONS):
        raise ValueError("complete support digest section count is malformed")
    identity_values = tuple(
        value for page in pages_by_section["input_match_identity_sha256"]
        for value in page.digests)
    expected_identity = tuple(_identity_digest([_row_identity(row)]) for row in rows)
    if identity_values != expected_identity:
        raise ValueError("complete support identity digest section does not replay")
    if root.whole_universe_identity_sha256 != complete_support_universe_digest(rows):
        raise ValueError("complete support whole-universe digest does not replay")
    kept = [_row_identity(row) for row in rows
            if row.disposition == plane_graph.SAME_AXIS_DISPOSITION_KEPT]
    if root.input_match_identity_sha256 != _identity_digest(kept):
        raise ValueError("complete support kept-subset identity digest does not replay")
    # Independent cap enforcement from the root's own declared plan. This is the
    # validation seam a directly constructed or tampered capture must pass, so it
    # re-measures every persisted page and refuses anything beyond the plan.
    for page in capture.index_pages:
        if len(_canonical_bytes(page.to_record())) > plan.index_page_max_bytes:
            raise ValueError("complete support index page exceeds the declared cap")
    for page in capture.digest_pages:
        if len(_canonical_bytes(page.to_record())) > plan.digest_page_max_bytes:
            raise ValueError("complete support digest page exceeds the declared cap")
    aggregate = 0
    for page in capture.sidecar_pages:
        size = len(_canonical_bytes(page.to_record()))
        if size > plan.sidecar_page_max_bytes:
            raise ValueError("complete support sidecar page exceeds the declared cap")
        aggregate += size
        for entry in page.entries:
            if entry.payload.payload_bytes() > plan.sidecar_entry_max_raw_bytes:
                raise ValueError(
                    "complete support sidecar entry exceeds the declared raw cap")
            if len(_canonical_bytes(entry.to_record())) > plan.sidecar_entry_max_bytes:
                raise ValueError(
                    "complete support sidecar entry exceeds the declared cap")
    if aggregate > plan.sidecar_aggregate_max_bytes:
        raise ValueError(
            "complete support sidecar aggregate exceeds the declared cap")


# --- Page packing -------------------------------------------------------------

def _page_size(payload: dict, digest_key: str) -> int:
    return len(_canonical_bytes({**payload, digest_key: _SYNTHETIC_DIGEST}))


def _max_fitting(size, upper: int, cap: int) -> int | None:
    """Largest ``k`` in ``[1, upper]`` with ``size(k) <= cap``, else ``None``.

    Galloping from 1 avoids probing near ``upper`` first. That matters because
    one worst-case sidecar entry already nearly fills a page, so a midpoint-first
    binary search would serialize thousands of entries per probe.
    """
    if upper < 1 or size(1) > cap:
        return None
    best, probe = 1, 2
    while probe < upper and size(probe) <= cap:
        best, probe = probe, min(upper, probe * 2)
    low, high = best + 1, min(probe, upper)
    while low <= high:
        middle = (low + high) // 2
        if size(middle) <= cap:
            best, low = middle, middle + 1
        else:
            high = middle - 1
    return best


def _index_page_size(rows: tuple, page_index: int, first: int) -> int:
    return _page_size(_index_page_payload(
        _IndexPageView(page_index, first, rows), include_digest=False), "page_sha256")


@dataclass(frozen=True, slots=True)
class _IndexPageView:
    page_index: int
    first_admitted_row_index: int
    rows: tuple
    schema: str = INDEX_PAGE_SCHEMA

    @property
    def page_sha256(self) -> str:
        return ""


def _make_index_page(rows: tuple, page_index: int, first: int) -> CandidateSupportIndexPageV2:
    digest = _digest(INDEX_PAGE_DIGEST_DOMAIN,
                     _index_page_payload(_IndexPageView(page_index, first, rows),
                                         include_digest=False))
    return CandidateSupportIndexPageV2(
        page_index=page_index, first_admitted_row_index=first, rows=tuple(rows),
        page_sha256=digest)


def _pack_index_pages(rows: tuple, cap: int) -> tuple[CandidateSupportIndexPageV2, ...]:
    pages: list[CandidateSupportIndexPageV2] = []
    start = 0
    while start < len(rows):
        best: int | None = None
        low, high = start + 1, len(rows)
        while low <= high:
            middle = (low + high) // 2
            if _index_page_size(rows[start:middle], len(pages), start) <= cap:
                best, low = middle, middle + 1
            else:
                high = middle - 1
        if best is None:
            raise CandidateSupportError("index_page_over_cap",
                                        f"index_page.{len(pages)}")
        pages.append(_make_index_page(rows[start:best], len(pages), start))
        start = best
    return tuple(pages)


def _sidecar_page_size(entries: tuple, page_index: int, first: int) -> int:
    return _page_size(_sidecar_page_payload(
        _SidecarPageView(page_index, first, entries), include_digest=False),
        "page_sha256")


@dataclass(frozen=True, slots=True)
class _SidecarPageView:
    page_index: int
    first_admitted_row_index: int
    entries: tuple
    schema: str = SIDECAR_PAGE_SCHEMA

    @property
    def page_sha256(self) -> str:
        return ""


def _make_sidecar_page(entries: tuple, page_index: int,
                       first: int) -> MatcherSupportSidecarPageV2:
    digest = _digest(SIDECAR_PAGE_DIGEST_DOMAIN, _sidecar_page_payload(
        _SidecarPageView(page_index, first, entries), include_digest=False))
    return MatcherSupportSidecarPageV2(
        page_index=page_index, first_admitted_row_index=first, entries=tuple(entries),
        page_sha256=digest)


def _pack_sidecar_pages(entries: tuple,
                        cap: int) -> tuple[MatcherSupportSidecarPageV2, ...]:
    """Pack present-row sidecar entries into bounded, ordered pages.

    ``entries`` is the present-row subsequence of the admitted universe, so its
    ordinals are **not** the admitted-row indices: a page's
    ``first_admitted_row_index`` is the admitted index of its own first entry
    (``entries[start].admitted_row_index``), never the slice ordinal ``start``.
    Every entry keeps its original admitted index, so a gap between present rows
    (or across a page boundary) is expected and is not renumbered.
    """
    pages: list[MatcherSupportSidecarPageV2] = []
    start = 0
    while start < len(entries):
        first = entries[start].admitted_row_index
        best: int | None = None
        low, high = start + 1, len(entries)
        while low <= high:
            middle = (low + high) // 2
            if _sidecar_page_size(entries[start:middle], len(pages), first) <= cap:
                best, low = middle, middle + 1
            else:
                high = middle - 1
        if best is None:
            raise CandidateSupportError("sidecar_page_over_cap",
                                        f"sidecar_page.{len(pages)}")
        pages.append(_make_sidecar_page(entries[start:best], len(pages), first))
        start = best
    return tuple(pages)


def _digest_page_size(section: str, page_index: int, first: int,
                      digests: tuple) -> int:
    return _page_size(_digest_page_payload(
        _DigestPageView(section, page_index, first, digests), include_digest=False),
        "page_sha256")


@dataclass(frozen=True, slots=True)
class _DigestPageView:
    section: str
    page_index: int
    first_admitted_row_index: int
    digests: tuple
    schema: str = DIGEST_PAGE_SCHEMA

    @property
    def page_sha256(self) -> str:
        return ""


def _make_digest_page(section: str, page_index: int, first: int,
                      digests: tuple) -> CandidateSupportDigestPageV2:
    digest = _digest(DIGEST_PAGE_DIGEST_DOMAIN, _digest_page_payload(
        _DigestPageView(section, page_index, first, digests), include_digest=False))
    return CandidateSupportDigestPageV2(
        section=section, page_index=page_index, first_admitted_row_index=first,
        digests=tuple(digests), page_sha256=digest)


def _pack_digest_pages(sections, cap: int) -> tuple[CandidateSupportDigestPageV2, ...]:
    pages: list[CandidateSupportDigestPageV2] = []
    for section in DIGEST_SECTIONS:
        values = tuple(sections[section])
        start = 0
        page_index = 0
        while start < len(values):
            best: int | None = None
            low, high = start + 1, len(values)
            while low <= high:
                middle = (low + high) // 2
                if _digest_page_size(section, page_index, start,
                                     values[start:middle]) <= cap:
                    best, low = middle, middle + 1
                else:
                    high = middle - 1
            if best is None:
                raise CandidateSupportError(
                    "digest_page_over_cap", f"digest_page.{section}.{page_index}")
            pages.append(_make_digest_page(section, page_index, start,
                                           values[start:best]))
            start = best
            page_index += 1
    return tuple(pages)


# --- Assembly -----------------------------------------------------------------

def _assemble_complete_support(
    *,
    rows: tuple,
    digest_sections,
    sidecar_entries: tuple,
    failed_assignment_sha256: str,
    expected_row_count: int | None,
    expected_whole_universe_identity_sha256: str | None,
    expected_input_match_identity_sha256: str | None,
    ingress: tuple,
    plan: "CompleteSupportCapPlanV1",
) -> CompleteSupportCaptureV2:
    if not isinstance(rows, tuple) or not rows:
        raise CandidateSupportError("universe_count_mismatch", "row_count")
    if type(expected_row_count) is not int or expected_row_count <= 0:
        raise CandidateSupportError("universe_count_mismatch", "expected_row_count")
    if len(rows) != expected_row_count:
        raise CandidateSupportError("universe_count_mismatch", "row_count")
    if len(rows) > plan.universe_max_rows:
        raise CandidateSupportError("row_cap_exceeded", "row_count")
    for position, row in enumerate(rows):
        if not isinstance(row, CandidateSupportIndexRowV2):
            raise CandidateSupportError("universe_order_mismatch", "row")
        if row.admitted_row_index != position:
            raise CandidateSupportError("universe_order_mismatch", "admitted_row_index")
        over = _row_field_over_cap(row)
        if over:
            raise CandidateSupportError("index_row_over_cap", f"row.{position}.{over}")
    if set(digest_sections) != set(DIGEST_SECTIONS):
        raise CandidateSupportError("universe_identity_mismatch", "digest_sections")
    for section in DIGEST_SECTIONS:
        values = tuple(digest_sections[section])
        if len(values) != len(rows):
            raise CandidateSupportError("universe_identity_mismatch",
                                        f"digest_sections.{section}")
        if any(value is not None and not _hex64(value) for value in values):
            raise CandidateSupportError("universe_identity_mismatch",
                                        f"digest_sections.{section}")
    expected_identity = tuple(_identity_digest([_row_identity(row)]) for row in rows)
    if tuple(digest_sections["input_match_identity_sha256"]) != expected_identity:
        raise CandidateSupportError("universe_identity_mismatch",
                                    "input_match_identity_sha256")
    if not _hex64(failed_assignment_sha256):
        raise CandidateSupportError("universe_identity_mismatch",
                                    "failed_assignment_sha256")
    if len(ingress) > INGRESS_MAX_RECORDS:
        raise CandidateSupportError("ingress_over_cap", "ingress_count")
    for position, record in enumerate(ingress):
        if not isinstance(record, FrameIngressProvenanceV1):
            raise CandidateSupportError("universe_identity_mismatch", "ingress")
        over = _ingress_record_over_cap(record)
        if over:
            raise CandidateSupportError("ingress_over_cap",
                                        f"ingress.{position}.{over}")
    whole_universe = complete_support_universe_digest(rows)
    if (expected_whole_universe_identity_sha256
            and whole_universe != expected_whole_universe_identity_sha256):
        raise CandidateSupportError("universe_identity_mismatch",
                                    "whole_universe_identity_sha256")
    kept_identity = _identity_digest(
        [_row_identity(row) for row in rows
         if row.disposition == plane_graph.SAME_AXIS_DISPOSITION_KEPT])
    if (expected_input_match_identity_sha256
            and kept_identity != expected_input_match_identity_sha256):
        raise CandidateSupportError("universe_identity_mismatch",
                                    "input_match_identity_sha256")
    present = tuple(row for row in rows if row.support_state == SUPPORT_STATE_PRESENT)
    if len(sidecar_entries) != len(present):
        raise CandidateSupportError("universe_identity_mismatch", "sidecar_entries")
    for row, entry in zip(present, sidecar_entries):
        if not isinstance(entry, MatcherSupportSidecarEntryV2):
            raise CandidateSupportError("universe_order_mismatch", "sidecar_entry")
        if entry.admitted_row_index != row.admitted_row_index:
            raise CandidateSupportError("universe_order_mismatch", "sidecar_entry")
        if entry.payload_sha256 != row.support_payload_sha256:
            raise CandidateSupportError("universe_identity_mismatch",
                                        "support_payload_sha256")
    index_pages = _pack_index_pages(rows, plan.index_page_max_bytes)
    digest_pages = _pack_digest_pages(digest_sections, plan.digest_page_max_bytes)
    sidecar_pages = _pack_sidecar_pages(sidecar_entries, plan.sidecar_page_max_bytes)
    aggregate = sum(len(_canonical_bytes(page.to_record())) for page in sidecar_pages)
    if aggregate > plan.sidecar_aggregate_max_bytes:
        raise CandidateSupportError("sidecar_aggregate_over_cap", "sidecar_aggregate_bytes")
    index_bindings = tuple(
        CandidateSupportIndexPageBindingV2(page.page_index, page.page_sha256,
                                           page.row_count) for page in index_pages)
    sidecar_bindings = tuple(
        CandidateSupportSidecarPageBindingV2(page.page_index, page.page_sha256,
                                             page.entry_count) for page in sidecar_pages)
    digest_bindings = tuple(
        _digest_section_from_pages(
            section, tuple(page for page in digest_pages if page.section == section))
        for section in DIGEST_SECTIONS)
    root = CandidateSupportCaptureRootV2(
        status="complete", basis=COMPLETE_SUPPORT_BASIS, reason=None, field=None,
        failed_assignment_sha256=failed_assignment_sha256,
        whole_universe_identity_sha256=whole_universe,
        input_match_identity_sha256=kept_identity, row_count=len(rows),
        declared_universe_cap=plan.universe_max_rows, index_pages=index_bindings,
        sidecar_pages=sidecar_bindings, digest_sections=digest_bindings,
        ingress=ingress, cap_plan=plan)
    if len(_canonical_bytes(root.to_record())) > plan.root_max_bytes:
        raise CandidateSupportError("root_over_cap", "root_bytes")
    return CompleteSupportCaptureV2(
        root=root, index_pages=index_pages, digest_pages=digest_pages,
        sidecar_pages=sidecar_pages)


@dataclass(frozen=True, slots=True)
class CompleteSupportCapPlanV1:
    """One declared cap plan; every value here is a declared, unapproved cap."""

    index_page_max_bytes: int = INDEX_PAGE_MAX_BYTES
    digest_page_max_bytes: int = DIGEST_PAGE_MAX_BYTES
    sidecar_page_max_bytes: int = SIDECAR_PAGE_MAX_BYTES
    sidecar_entry_max_raw_bytes: int = SIDECAR_ENTRY_MAX_RAW_BYTES
    sidecar_entry_max_bytes: int = SIDECAR_ENTRY_MAX_BYTES
    sidecar_aggregate_max_bytes: int = SIDECAR_AGGREGATE_MAX_BYTES
    root_max_bytes: int = ROOT_MAX_BYTES
    universe_max_rows: int = UNIVERSE_MAX_ROWS

    def __post_init__(self) -> None:
        for name in CAP_PLAN_FIELDS:
            value = getattr(self, name)
            if (type(value) is not int or value <= 0
                    or value > CAP_PLAN_MAX_ABS_INT):
                raise ValueError(
                    f"complete support cap {name} must be a positive bounded int")

    def as_record(self) -> dict:
        return {name: getattr(self, name) for name in CAP_PLAN_FIELDS}


def _cap_plan_field_over_cap(plan: object) -> str:
    """Return the first invalid plan field name, or ``""`` when it is legal.

    Validation is value-based and independent of construction, because a frozen
    dataclass can be mutated afterwards with ``object.__setattr__``; a
    construction-time check alone is therefore not tamper protection.

    Every consumer enforces the **live attributes** (``plan.<field>``), so those
    are what this predicate validates. It deliberately does not trust the
    overridable :meth:`~CompleteSupportCapPlanV1.as_record` mapping: a subclass
    can return a default-valid record while carrying invalid live caps (for
    example zero sidecar caps), which would otherwise let a re-signed capture
    assemble or validate under values the runtime never uses. The exact plan class
    is therefore required, and the canonical record must additionally agree with
    the live attributes and carry exactly :data:`CAP_PLAN_FIELDS` (no missing or
    extra serialized field). Each live value must be an exact positive ``int`` at
    most ``CAP_PLAN_MAX_ABS_INT``.
    """
    # ``type(...) is`` rather than ``isinstance(...)``: a subclass can override
    # ``as_record`` (a valid-looking mapping) while its real cap attributes are
    # invalid, and consumers read those live attributes.
    if type(plan) is not CompleteSupportCapPlanV1:
        return "plan"
    for name in CAP_PLAN_FIELDS:
        try:
            value = getattr(plan, name)
        except AttributeError:
            return "plan"
        if (type(value) is not int or value <= 0
                or value > CAP_PLAN_MAX_ABS_INT):
            return name
    try:
        record = plan.as_record()
    except (AttributeError, TypeError, ValueError):
        return "plan"
    if type(record) is not dict or set(record) != set(CAP_PLAN_FIELDS):
        return "plan"
    for name in CAP_PLAN_FIELDS:
        # The serialized record must agree with the live attribute that runtime
        # enforcement and the cap-plan digest actually bind.
        if record[name] != getattr(plan, name):
            return name
    return ""


def _resolve_plan(plan: object) -> "CompleteSupportCapPlanV1 | None":
    """Resolve the caller's plan, or ``None`` when plan resolution itself fails.

    ``None`` (the default plan) is the source default. Any object that is not a
    value-valid :class:`CompleteSupportCapPlanV1` (including a frozen plan mutated
    with ``object.__setattr__``, or a default plan whose own construction fails) is
    a plan-resolution failure and must produce the separately bounded planless
    refusal instead of the planned one.
    """
    if plan is None:
        try:
            candidate: object = CompleteSupportCapPlanV1()
        except (TypeError, ValueError, OverflowError):
            return None
    elif isinstance(plan, CompleteSupportCapPlanV1):
        candidate = plan
    else:
        return None
    if (isinstance(candidate, CompleteSupportCapPlanV1)
            and not _cap_plan_field_over_cap(candidate)):
        return candidate
    return None


# --- Frozen Route B diagnostic profile ---------------------------------------
#
# The lead accepted this exact single-run profile on 2026-10-04 for the frozen
# 1,928-row ordinary-image Route B diagnostic universe. It is a fixed-run
# diagnostic profile, not a general production cap: it changes no default and no
# constant above, and it authorizes no capture by itself (the graph seam's
# ``complete_support_sink`` must be explicitly attached).
ROUTE_B_DIAGNOSTIC_EXPECTED_ROW_COUNT = 1_928
ROUTE_B_DIAGNOSTIC_CAP_PLAN = CompleteSupportCapPlanV1(
    index_page_max_bytes=262_144,
    digest_page_max_bytes=262_144,
    sidecar_page_max_bytes=237_568,
    sidecar_entry_max_raw_bytes=131_072,
    sidecar_entry_max_bytes=233_472,
    sidecar_aggregate_max_bytes=536_870_912,
    root_max_bytes=512_000,
    universe_max_rows=1_928)


def build_complete_support_capture(
    *,
    rows,
    digest_sections,
    sidecar_entries,
    failed_assignment_sha256: str = "",
    expected_row_count: int | None = None,
    expected_whole_universe_identity_sha256: str | None = None,
    expected_input_match_identity_sha256: str | None = None,
    ingress=(),
    plan: CompleteSupportCapPlanV1 | None = None,
) -> CompleteSupportCaptureV2:
    """Assemble one staged V2 capture, or refuse it atomically."""
    resolved_plan = _resolve_plan(plan)
    if resolved_plan is None:
        return refused_complete_support(PLANLESS_REFUSAL_REASON,
                                        PLANLESS_REFUSAL_FIELD)
    try:
        return _assemble_complete_support(
            rows=tuple(rows), digest_sections=digest_sections,
            sidecar_entries=tuple(sidecar_entries),
            failed_assignment_sha256=(failed_assignment_sha256 or _sha(
                {"domain": "failed_best_assignment", "payload": []})),
            expected_row_count=expected_row_count,
            expected_whole_universe_identity_sha256=(
                expected_whole_universe_identity_sha256),
            expected_input_match_identity_sha256=(
                expected_input_match_identity_sha256),
            ingress=tuple(ingress), plan=resolved_plan)
    except CandidateSupportError as error:
        return refused_complete_support(error.reason, error.field or "record",
                                        plan=resolved_plan)
    except (AttributeError, TypeError, ValueError, IndexError, KeyError,
            OverflowError, ZeroDivisionError) as error:
        # Fixed canonical built-in category only; a custom subclass name would be
        # caller-controlled text and must not reach the persisted refusal field.
        return refused_complete_support("universe_identity_mismatch",
                                        _raw_failure_exception_category(error),
                                        plan=resolved_plan)


# --- Collector over the shipped sinks ----------------------------------------

def _resolve_row_support(row, left_witness, right_witness, plan):
    """Return ``(support_state, refusal_reason, payload_or_None)`` for one row."""
    if row.payload is None:
        reason = row.refusal_reason
        if reason not in SUPPORT_REFUSAL_REASONS:
            reason = MATCHER_WITNESS_UNAVAILABLE
        return SUPPORT_STATE_REFUSED, reason, None
    if row.payload.envelope.matcher_witness_sha256 != row.payload.witness.witness_sha256:
        return SUPPORT_STATE_REFUSED, "witness_digest_mismatch", None
    try:
        payload = _support_payload(row)
    except CandidateSupportError as error:
        if error.field == "payload_sha256":
            return SUPPORT_STATE_REFUSED, "payload_digest_mismatch", None
        if error.reason == SUPPORT_PAYLOAD_OVER_CAP:
            return SUPPORT_STATE_REFUSED, "support_payload_over_cap", None
        return SUPPORT_STATE_REFUSED, "support_payload_unavailable", None
    except (AttributeError, TypeError, ValueError, IndexError, KeyError):
        return SUPPORT_STATE_REFUSED, "support_payload_unavailable", None
    if payload is None:
        return SUPPORT_STATE_REFUSED, "support_payload_unavailable", None
    if _payload_over_declared_bounds(payload):
        return SUPPORT_STATE_REFUSED, "support_payload_over_cap", None
    if payload.payload_bytes() > plan.sidecar_entry_max_raw_bytes:
        return SUPPORT_STATE_REFUSED, "support_payload_over_cap", None
    if left_witness is None or right_witness is None:
        return SUPPORT_STATE_REFUSED, "source_tap_witness_unavailable", None
    if left_witness.ingress.source_shape != right_witness.ingress.source_shape:
        return SUPPORT_STATE_REFUSED, "source_tap_witness_shape_mismatch", None
    entry = MatcherSupportSidecarEntryV2(
        admitted_row_index=row.admitted_row_index, payload=payload)
    if len(_canonical_bytes(entry.to_record())) > plan.sidecar_entry_max_bytes:
        return SUPPORT_STATE_REFUSED, "support_payload_over_cap", None
    return SUPPORT_STATE_PRESENT, None, payload


def collect_complete_candidate_support(
    readings: object,
    matches: object,
    signature_witnesses: object,
    matcher_witness_rows: object,
    same_axis_observations: object,
    *,
    ingress=(),
    failed_assignment_sha256: str = "",
    expected_row_count: int | None = None,
    expected_whole_universe_identity_sha256: str | None = None,
    expected_input_match_identity_sha256: str | None = None,
    plan: CompleteSupportCapPlanV1 | None = None,
) -> CompleteSupportCaptureV2:
    """Collect the complete V2 capture from the shipped sinks, default-off.

    It re-matches, re-signatures, re-solves and re-reads nothing; it walks the
    same observation/witness-row material V1 consumes and applies the design's
    row-local typed support refusal.
    """
    resolved_plan = _resolve_plan(plan)
    if resolved_plan is None:
        return refused_complete_support(PLANLESS_REFUSAL_REASON,
                                        PLANLESS_REFUSAL_FIELD)
    failure_site = _CollectorFailureSite()
    try:
        return _collect_complete_support(
            readings, matches, signature_witnesses, matcher_witness_rows,
            same_axis_observations, ingress=tuple(ingress),
            failed_assignment_sha256=failed_assignment_sha256,
            expected_row_count=expected_row_count,
            expected_whole_universe_identity_sha256=(
                expected_whole_universe_identity_sha256),
            expected_input_match_identity_sha256=(
                expected_input_match_identity_sha256),
            plan=resolved_plan, failure_site=failure_site)
    except CandidateSupportError as error:
        return refused_complete_support(error.reason, error.field or "record",
                                        plan=resolved_plan)
    except (AttributeError, TypeError, ValueError, IndexError, KeyError,
            OverflowError, ZeroDivisionError) as error:
        # Diagnostic-only localization: the reason stays the fixed compatibility
        # code, while the field names a fixed canonical built-in category plus the
        # innermost module/operation/line and, for a row-local raw exception, the
        # exact admitted position (never a custom class name or error text).
        return refused_complete_support(
            "universe_identity_mismatch",
            _raw_failure_field(error, failure_site.position),
            plan=resolved_plan)


def _collect_complete_support(
    readings, matches, signature_witnesses, matcher_witness_rows,
    same_axis_observations, *, ingress, failed_assignment_sha256,
    expected_row_count, expected_whole_universe_identity_sha256,
    expected_input_match_identity_sha256, plan, failure_site=None,
) -> CompleteSupportCaptureV2:
    if not isinstance(same_axis_observations, plane_graph.SameAxisCandidateObservation):
        raise CandidateSupportError("universe_count_mismatch", "same_axis_observation")
    observations = same_axis_observations
    if type(expected_row_count) is not int or expected_row_count <= 0:
        raise CandidateSupportError("universe_count_mismatch", "expected_row_count")
    if not isinstance(matcher_witness_rows, (list, tuple)):
        raise CandidateSupportError("universe_count_mismatch", "matcher_witness_rows")
    if len(observations.dispositions) != len(observations.candidates):
        raise CandidateSupportError("universe_count_mismatch", "dispositions")
    if len(matcher_witness_rows) != len(observations.candidates):
        raise CandidateSupportError("universe_count_mismatch", "matcher_witness_rows")
    if not observations.candidates:
        raise CandidateSupportError("universe_count_mismatch", "candidates")
    if len(observations.candidates) != expected_row_count:
        raise CandidateSupportError("universe_count_mismatch", "row_count")
    if len(observations.candidates) > plan.universe_max_rows:
        raise CandidateSupportError("row_cap_exceeded", "row_count")
    witness_by_signature: dict[str, SourceTapWitnessBindingV1] = {}
    for item in signature_witnesses or ():
        if isinstance(item, SourceTapWitnessBindingV1):
            witness_by_signature[item.signature_sha256] = item
    kept_positions = {id(match): index for index, match in enumerate(matches)}
    groups: dict[tuple, int] = {}
    rows: list[CandidateSupportIndexRowV2] = []
    entries: list[MatcherSupportSidecarEntryV2] = []
    sections: dict[str, list] = {section: [] for section in DIGEST_SECTIONS}
    for position, (candidate, disposition) in enumerate(zip(
            observations.candidates, observations.dispositions)):
        if failure_site is not None:
            failure_site.enter_row(position)
        row = matcher_witness_rows[position]
        if not isinstance(row, plane_graph.PreFilterMatcherWitnessRowV1):
            raise CandidateSupportError("universe_order_mismatch", "matcher_witness_row")
        if row.admitted_row_index != position or row.disposition != disposition:
            raise CandidateSupportError("universe_order_mismatch",
                                        "row_disposition_binding")
        group_key = (candidate.left, candidate.right, candidate.left_axis,
                     candidate.quadrant_of_right)
        group_index = groups.setdefault(group_key, len(groups))
        match_index = (kept_positions.get(id(candidate))
                       if disposition == plane_graph.SAME_AXIS_DISPOSITION_KEPT
                       else None)
        left_witness = witness_by_signature.get(row.left_signature_sha256 or "")
        right_witness = witness_by_signature.get(row.right_signature_sha256 or "")
        state, reason, support = _resolve_row_support(
            row, left_witness, right_witness, plan)
        index_row = CandidateSupportIndexRowV2(
            admitted_row_index=position, group_index=group_index,
            left=candidate.left, right=candidate.right,
            left_axis=int(candidate.left_axis), right_axis=int(candidate.right_axis),
            quadrant_of_right=int(candidate.quadrant_of_right),
            left_plane=int(candidate.left_plane),
            right_plane=int(candidate.right_plane),
            left_alternative_index=_alternative_index(
                readings, candidate.left, candidate.left_plane, candidate.left_b),
            right_alternative_index=_alternative_index(
                readings, candidate.right, candidate.right_plane, candidate.right_b),
            left_b=float(candidate.left_b), right_b=float(candidate.right_b),
            delta=_delta(candidate, readings), tolerance=CANDIDATE_SUPPORT_TOLERANCE,
            disposition=disposition, match_index=match_index,
            overlap_cells=int(candidate.overlap_cells),
            cell_shift_rc=(None if candidate.cell_shift_rc is None
                           else (int(candidate.cell_shift_rc[0]),
                                 int(candidate.cell_shift_rc[1]))),
            support_state=state, support_refusal_reason=reason,
            support_payload_sha256=(None if support is None
                                    else support.payload_sha256))
        rows.append(index_row)
        if support is not None:
            entries.append(MatcherSupportSidecarEntryV2(
                admitted_row_index=position, payload=support))
        sections["match_sha256"].append(row.match_sha256)
        sections["envelope_sha256"].append(row.envelope_sha256)
        sections["left_signature_sha256"].append(row.left_signature_sha256)
        sections["right_signature_sha256"].append(row.right_signature_sha256)
        sections["matcher_witness_sha256"].append(
            None if row.payload is None else row.payload.witness.witness_sha256)
        sections["input_match_identity_sha256"].append(
            _identity_digest([_row_identity(index_row)]))
    if failure_site is not None:
        failure_site.leave_row()
    return _assemble_complete_support(
        rows=tuple(rows), digest_sections=sections,
        sidecar_entries=tuple(entries),
        failed_assignment_sha256=(failed_assignment_sha256 or _sha(
            {"domain": "failed_best_assignment", "payload": []})),
        expected_row_count=expected_row_count,
        expected_whole_universe_identity_sha256=(
            expected_whole_universe_identity_sha256),
        expected_input_match_identity_sha256=(
            expected_input_match_identity_sha256),
        ingress=ingress, plan=plan)


def _match_order_identity(matches) -> str:
    """Ordered kept-subset identity digest over the returned match tuple.

    The shipped identity is rebuilt from the collector's own candidate walk;
    deriving it independently here from the exact post-filter match order binds
    the capture to the match tuple the solver actually consumed. A foreign,
    dropped or reordered kept match then becomes an atomic
    ``universe_identity_mismatch`` rather than a silently different kept-subset
    digest.
    """
    identities = []
    for match in matches:
        identities.append([
            match.left, match.left_plane, match.left_axis, match.right,
            match.right_plane, match.right_axis, match.quadrant_of_right,
            float(match.left_b), float(match.right_b)])
    return _identity_digest(identities)


def collect_route_b_diagnostic_support(
        readings, matches, signature_witnesses, matcher_witness_rows,
        same_axis_observations, *, ingress=(),
        failed_assignment_sha256: str = "",
        expected_row_count: int = ROUTE_B_DIAGNOSTIC_EXPECTED_ROW_COUNT,
        expected_whole_universe_identity_sha256: str | None = None,
        plan: CompleteSupportCapPlanV1 = ROUTE_B_DIAGNOSTIC_CAP_PLAN,
) -> CompleteSupportCaptureV2:
    """Collect the complete V2 capture for the frozen Route B diagnostic.

    This is the exact-profile default the graph seam calls when its
    ``complete_support_sink`` is attached: the accepted single-run cap profile
    plus the exact frozen 1,928-row expectation. ``matches`` is the solver's own
    returned post-filter tuple, whose order is independently bound as the
    capture's ``input_match_identity_sha256``, so a kept-subset mismatch refuses
    atomically. Every other contract - row-local typed support refusal over all
    admitted rows, order/count/identity binding, the capture-global byte caps
    and the all-or-nothing page staging - is the unchanged collector above. It
    re-matches, re-signatures and re-solves nothing.
    """
    try:
        expected_input_match_identity_sha256 = _match_order_identity(matches)
    except (AttributeError, TypeError, ValueError, IndexError, KeyError):
        expected_input_match_identity_sha256 = None
    return collect_complete_candidate_support(
        readings, matches, signature_witnesses, matcher_witness_rows,
        same_axis_observations, ingress=ingress,
        failed_assignment_sha256=failed_assignment_sha256,
        expected_row_count=expected_row_count,
        expected_whole_universe_identity_sha256=(
            expected_whole_universe_identity_sha256),
        expected_input_match_identity_sha256=(
            expected_input_match_identity_sha256),
        plan=plan)


# --- Deterministic synthetic worst-case preflight -----------------------------

@dataclass(frozen=True, slots=True)
class CompleteSupportPreflightV1:
    """Measured canonical sizes, derived page counts and proposed caps."""

    preflight_status: str
    preflight_refusal_reason: str
    declared_universe_cap: int
    cap_invariant_ok: bool
    measured_bytes: tuple[tuple[str, int], ...]
    derived_counts: tuple[tuple[str, int], ...]
    proposed_caps: tuple[tuple[str, int], ...]
    detail: str = ""
    schema: str = PREFLIGHT_SCHEMA

    def measurement(self, name: str) -> int | None:
        return dict(self.measured_bytes).get(name)

    def derived(self, name: str) -> int | None:
        return dict(self.derived_counts).get(name)

    def proposed_cap(self, name: str) -> int | None:
        return dict(self.proposed_caps).get(name)

    def to_record(self) -> dict:
        return {"cap_invariant_ok": self.cap_invariant_ok,
                "declared_universe_cap": self.declared_universe_cap,
                "derived_counts": {name: value
                                   for name, value in self.derived_counts},
                "detail": self.detail,
                "measured_bytes": {name: value
                                   for name, value in self.measured_bytes},
                "preflight_refusal_reason": self.preflight_refusal_reason,
                "preflight_status": self.preflight_status,
                "proposed_caps": {name: value
                                  for name, value in self.proposed_caps},
                "schema": self.schema}


_SYNTHETIC_INDEX_ROW_CACHE: dict[int, CandidateSupportIndexRowV2] = {}


def _synthetic_index_row(universe_cap: int) -> CandidateSupportIndexRowV2:
    cached = _SYNTHETIC_INDEX_ROW_CACHE.get(universe_cap)
    if cached is not None:
        return cached
    widest = "z" * INDEX_ROW_MAX_FRAME_ID_BYTES
    # ``-INDEX_ROW_MAX_ABS_INT`` is the longest declared-bounded canonical
    # integer; the non-negative-only fields use the positive bound.
    row = CandidateSupportIndexRowV2(
        admitted_row_index=universe_cap - 1, group_index=INDEX_ROW_MAX_ABS_INT,
        left=widest, right=widest, left_axis=-INDEX_ROW_MAX_ABS_INT,
        right_axis=-INDEX_ROW_MAX_ABS_INT,
        quadrant_of_right=-INDEX_ROW_MAX_ABS_INT,
        left_plane=-INDEX_ROW_MAX_ABS_INT, right_plane=-INDEX_ROW_MAX_ABS_INT,
        left_alternative_index=INDEX_ROW_MAX_ABS_INT,
        right_alternative_index=INDEX_ROW_MAX_ABS_INT,
        left_b=_SYNTHETIC_MAX_FLOAT, right_b=_SYNTHETIC_MAX_FLOAT,
        delta=_SYNTHETIC_MAX_FLOAT, tolerance=_SYNTHETIC_MAX_FLOAT,
        disposition=plane_graph.SAME_AXIS_DISPOSITION_KEPT,
        match_index=INDEX_ROW_MAX_ABS_INT, overlap_cells=INDEX_ROW_MAX_ABS_INT,
        cell_shift_rc=(-INDEX_ROW_MAX_ABS_INT, -INDEX_ROW_MAX_ABS_INT),
        support_state=SUPPORT_STATE_PRESENT, support_refusal_reason=None,
        support_payload_sha256=_SYNTHETIC_DIGEST)
    _SYNTHETIC_INDEX_ROW_CACHE[universe_cap] = row
    return row


def _synthetic_ingress_record() -> FrameIngressProvenanceV1:
    """The declared-maximal ingress pin, built as a real V1 record.

    Every field is at the exact bound enforced by :func:`_ingress_record_over_cap`:
    the widest frame/source strings, ``byte_length`` at ``INGRESS_MAX_ABS_INT``,
    the three-dimensional decoded shape the RGB decoder may produce (the third
    fixed channel value is the largest legal arity), two positive dimensions at
    ``INGRESS_MAX_SHAPE_DIM`` for every size tuple and the longest signed
    ``crop_origin`` coordinates.
    """
    dim = INGRESS_MAX_SHAPE_DIM
    return FrameIngressProvenanceV1(
        frame_id="z" * INGRESS_MAX_FRAME_ID_BYTES,
        source_label="z" * INGRESS_MAX_SOURCE_LABEL_BYTES,
        byte_sha256=_SYNTHETIC_DIGEST, byte_length=INGRESS_MAX_ABS_INT,
        decoded_shape=(dim, dim, 3), decoded_dtype="|u1",
        decoded_pixel_sha256=_SYNTHETIC_DIGEST,
        transform=FrameTransform(
            source_size=(dim, dim), crop_origin=(-INGRESS_MAX_ABS_INT,
                                                 -INGRESS_MAX_ABS_INT),
            crop_size=(dim, dim), output_size=(dim, dim)))


def _synthetic_sidecar_entry(raw_cap: int,
                             admitted_row_index: int) -> MatcherSupportSidecarEntryV2:
    """The proven maximum canonical sidecar entry under ``raw_cap``.

    The raw gate bounds the total element count to ``raw_cap // 8`` and every
    serialized element is declared-bounded by ``PAYLOAD_MAX_ABS_ELEMENT`` (the
    longest canonical value is the 11-character ``-2147483647``). Two effects
    decide the maximum: warp elements dominate logical-pair elements (whose
    values are additionally bounded by the source shape), and ``ndarray.tolist``
    emits one nested JSON list per array row, so a shape with more rows encodes
    more brackets and commas for the same element count. The maximum therefore
    spends the whole raw budget on the two warp arrays shaped ``(cells, 1)`` with
    the longest canonical value, and uses the largest declared shape dimension
    for ``source_shape``. No finite shape sample is used, so no smaller encoding
    can exceed this one. ``admitted_row_index`` is serialized, so the entry is
    built at the declared maximum admitted index to include its longest decimal
    form.
    """
    element_budget = raw_cap // 8
    warp_cells = element_budget // 2
    if warp_cells < 1:
        raise ValueError("sidecar raw cap cannot hold one warp cell")
    max_dim = PAYLOAD_MAX_SHAPE_DIM
    if warp_cells > max_dim:
        # The declared raw cap cannot be represented as a ``(cells, 1)`` grid;
        # the preflight marks such a plan ``cap_configuration_invalid`` rather
        # than claiming this clamped construction is the maximum.
        target_shape = (max_dim, max(1, warp_cells // max_dim))
    else:
        target_shape = (warp_cells, 1)
    rows_side, columns_side = target_shape
    warp_elements = rows_side * columns_side
    pair_count = max(0, (element_budget - 2 * warp_elements) // 4)
    logical = np.zeros((pair_count, 4), dtype=np.int64)
    if pair_count:
        logical[:, 0] = max_dim - 1
        logical[:, 1] = max_dim - 1
        logical[:, 2] = rows_side - 1
        logical[:, 3] = columns_side - 1
    warp_value = -PAYLOAD_MAX_ABS_ELEMENT
    warp_rows = np.full((rows_side, columns_side), warp_value, dtype=np.int64)
    warp_columns = np.full((rows_side, columns_side), warp_value, dtype=np.int64)
    for array in (logical, warp_rows, warp_columns):
        array.flags.writeable = False
    payload = MatcherSupportPayloadV1(
        logical_pairs=logical, warp_source_rows=warp_rows,
        warp_source_columns=warp_columns,
        source_shape=(max_dim, max_dim), target_shape=target_shape)
    return MatcherSupportSidecarEntryV2(
        admitted_row_index=admitted_row_index, payload=payload)


def _synthetic_root(index_count: int, sidecar_count: int, rows_per_index: int,
                    entries_per_sidecar: int, digests_per_page: int,
                    universe_cap: int, plan: CompleteSupportCapPlanV1,
                    ingress: tuple[FrameIngressProvenanceV1, ...]
                    ) -> CandidateSupportCaptureRootV2:
    index_bindings = tuple(
        CandidateSupportIndexPageBindingV2(index, _SYNTHETIC_DIGEST, rows_per_index)
        for index in range(index_count))
    sidecar_bindings = tuple(
        CandidateSupportSidecarPageBindingV2(index, _SYNTHETIC_DIGEST,
                                             entries_per_sidecar)
        for index in range(sidecar_count))
    sections = []
    for section in DIGEST_SECTIONS:
        pages = tuple(
            CandidateSupportDigestPageBindingV2(
                index, index * digests_per_page, digests_per_page,
                _SYNTHETIC_DIGEST)
            for index in range((universe_cap + digests_per_page - 1)
                               // digests_per_page))
        section_sha = _digest(DIGEST_SECTION_DIGEST_DOMAIN, {
            "pages": [page.sha256 for page in pages], "section": section})
        sections.append(CandidateSupportDigestSectionV2(
            section=section, page_count=len(pages), section_sha256=section_sha,
            pages=pages))
    return CandidateSupportCaptureRootV2(
        status="complete", basis=COMPLETE_SUPPORT_BASIS, reason=None, field=None,
        failed_assignment_sha256=_SYNTHETIC_DIGEST,
        whole_universe_identity_sha256=_SYNTHETIC_DIGEST,
        input_match_identity_sha256=_SYNTHETIC_DIGEST, row_count=universe_cap,
        declared_universe_cap=universe_cap, index_pages=index_bindings,
        sidecar_pages=sidecar_bindings, digest_sections=tuple(sections),
        ingress=ingress, cap_plan=plan)


def preflight_complete_support(
    plan: CompleteSupportCapPlanV1 | None = None,
) -> CompleteSupportPreflightV1:
    """Measure the synthetic worst case at the declared ceiling and gate it.

    Reads no candidate, witness or payload material: every shape is synthetic and
    every size is measured from canonical encodings. The declared plan is never
    silently adjusted; proposed caps are reported separately for the lead.
    """
    resolved = plan or CompleteSupportCapPlanV1()
    cap = resolved.universe_max_rows
    index_row = _synthetic_index_row(cap)
    index_row_bytes = len(_canonical_bytes(index_row.to_record()))
    # Worst-case page envelopes use the largest admitted index on every page.
    index_envelope = _index_page_size((index_row,), cap - 1, cap - 1) - index_row_bytes
    max_index_rows = _max_fitting(
        lambda count: _index_page_size((index_row,) * count, cap - 1, cap - 1),
        cap, resolved.index_page_max_bytes)
    index_page_bytes = (0 if max_index_rows is None else _index_page_size(
        (index_row,) * max_index_rows, cap - 1, cap - 1))
    index_page_count = (0 if not max_index_rows
                        else (cap + max_index_rows - 1) // max_index_rows)

    # The longest section name is the worst case for both the digest page
    # envelope and the number of digest pages (fewer digests fit per page, so the
    # root carries more bindings).
    longest_section = max(DIGEST_SECTIONS, key=len)
    digest_text = '"' + _SYNTHETIC_DIGEST + '"'
    digest_envelope = _page_size(_digest_page_payload(
        _DigestPageView(longest_section, cap - 1, cap - 1, ()),
        include_digest=False), "page_sha256")
    max_digests = _max_fitting(
        lambda count: _digest_page_size(longest_section, cap - 1, cap - 1,
                                        (digest_text,) * count),
        cap, resolved.digest_page_max_bytes)
    digest_page_bytes = (0 if max_digests is None else _digest_page_size(
        longest_section, cap - 1, cap - 1, (digest_text,) * max_digests))
    digest_page_count = 0 if not max_digests else 6 * (
        (cap + max_digests - 1) // max_digests)

    entry = _synthetic_sidecar_entry(resolved.sidecar_entry_max_raw_bytes, cap - 1)
    entry_bytes = len(_canonical_bytes(entry.to_record()))
    sidecar_envelope = _sidecar_page_size((entry,), cap - 1, cap - 1) - entry_bytes
    max_entries = _max_fitting(
        lambda count: _sidecar_page_size((entry,) * count, cap - 1, cap - 1),
        cap, resolved.sidecar_page_max_bytes)
    sidecar_page_bytes = (0 if max_entries is None else _sidecar_page_size(
        (entry,) * max_entries, cap - 1, cap - 1))
    sidecar_page_count = (0 if not max_entries
                          else (cap + max_entries - 1) // max_entries)
    sidecar_aggregate = sidecar_page_count * sidecar_page_bytes

    ingress_record = _synthetic_ingress_record()
    ingress_record_bytes = len(_canonical_bytes(ingress_record.to_record()))
    ingress_tuple = (ingress_record,) * INGRESS_MAX_RECORDS
    ingress_tuple_bytes = len(_canonical_bytes(
        [record.to_record() for record in ingress_tuple]))

    root_bytes = 0
    if max_index_rows and max_digests and max_entries:
        root = _synthetic_root(index_page_count, sidecar_page_count, max_index_rows,
                               max_entries, max_digests, cap, resolved,
                               ingress_tuple)
        root_bytes = len(_canonical_bytes(root.to_record()))

    # The maximal atomic refusal roots. A refusal that followed a resolved plan
    # preserves and hashes that plan and is bounded by the fixed ``ROOT_MAX_BYTES``
    # bound; the maximal planned refusal carries the longest reason code and a
    # refusal field at exactly the declared bound, and its maximal plan uses every
    # declared plan value at ``CAP_PLAN_MAX_ABS_INT``. A planless refusal (plan
    # resolution itself failed) is separately bounded by
    # ``PLANLESS_REFUSAL_ROOT_MAX_BYTES`` and is exactly the fixed
    # ``PLANLESS_REFUSAL_REASON``/``PLANLESS_REFUSAL_FIELD`` pair, so its maximal
    # form carries that fixed pair rather than an ordinary refusal code.
    longest_refusal_reason = max(CAPTURE_REFUSAL_REASONS, key=len)
    maximal_plan = CompleteSupportCapPlanV1(
        index_page_max_bytes=CAP_PLAN_MAX_ABS_INT,
        digest_page_max_bytes=CAP_PLAN_MAX_ABS_INT,
        sidecar_page_max_bytes=CAP_PLAN_MAX_ABS_INT,
        sidecar_entry_max_raw_bytes=CAP_PLAN_MAX_ABS_INT,
        sidecar_entry_max_bytes=CAP_PLAN_MAX_ABS_INT,
        sidecar_aggregate_max_bytes=CAP_PLAN_MAX_ABS_INT,
        root_max_bytes=CAP_PLAN_MAX_ABS_INT,
        universe_max_rows=CAP_PLAN_MAX_ABS_INT)
    refused_root = CandidateSupportCaptureRootV2(
        status="refused", basis=None, reason=longest_refusal_reason,
        field="z" * REFUSAL_FIELD_MAX_BYTES,
        failed_assignment_sha256="", whole_universe_identity_sha256="",
        input_match_identity_sha256="", row_count=0, declared_universe_cap=0,
        index_pages=(), sidecar_pages=(), digest_sections=(), ingress=(),
        cap_plan=maximal_plan)
    refused_root_bytes = len(_canonical_bytes(refused_root.to_record()))
    planless_refused_root = CandidateSupportCaptureRootV2(
        status="refused", basis=None, reason=PLANLESS_REFUSAL_REASON,
        field=PLANLESS_REFUSAL_FIELD,
        failed_assignment_sha256="", whole_universe_identity_sha256="",
        input_match_identity_sha256="", row_count=0, declared_universe_cap=0,
        index_pages=(), sidecar_pages=(), digest_sections=(), ingress=(),
        cap_plan=None)
    planless_refused_root_bytes = len(
        _canonical_bytes(planless_refused_root.to_record()))

    shape_representable = (
        1 <= resolved.sidecar_entry_max_raw_bytes // 16 <= PAYLOAD_MAX_SHAPE_DIM)
    invariant_ok = (shape_representable
                    and resolved.sidecar_entry_max_bytes
                    <= resolved.sidecar_page_max_bytes - sidecar_envelope)
    over: list[str] = []
    reason = ""
    if not invariant_ok:
        reason = PREFLIGHT_CAP_CONFIGURATION_INVALID
    elif not max_index_rows:
        reason = PREFLIGHT_SYNTHETIC_SIZE_OVER_CAP
        over.append("index_page")
    elif not max_digests:
        reason = PREFLIGHT_CAP_CONFIGURATION_INVALID
    elif not max_entries:
        reason = PREFLIGHT_AGGREGATE_UNBOUNDED
    else:
        if entry_bytes > resolved.sidecar_entry_max_bytes:
            over.append("sidecar_entry")
        if index_page_bytes > resolved.index_page_max_bytes:
            over.append("index_page")
        if digest_page_bytes > resolved.digest_page_max_bytes:
            over.append("digest_page")
        if sidecar_page_bytes > resolved.sidecar_page_max_bytes:
            over.append("sidecar_page")
        if root_bytes > resolved.root_max_bytes:
            over.append("root")
        if refused_root_bytes > ROOT_MAX_BYTES:
            over.append("refused_root")
        if planless_refused_root_bytes > PLANLESS_REFUSAL_ROOT_MAX_BYTES:
            over.append("planless_refused_root")
        if sidecar_aggregate > resolved.sidecar_aggregate_max_bytes:
            over.append("sidecar_aggregate")
        if over:
            reason = PREFLIGHT_SYNTHETIC_SIZE_OVER_CAP
    status = PREFLIGHT_OK if not reason else PREFLIGHT_REFUSED

    proposed_entry = _ceil_to(max(entry_bytes, 1), 4_096)
    proposed_sidecar_page = _ceil_to(
        max(sidecar_page_bytes, proposed_entry + sidecar_envelope, 4_096), 4_096)
    proposed = (
        ("INDEX_PAGE_MAX_BYTES", _ceil_to(max(index_page_bytes, 4_096), 4_096)),
        ("DIGEST_PAGE_MAX_BYTES", _ceil_to(max(digest_page_bytes, 4_096), 4_096)),
        ("SIDECAR_PAGE_MAX_BYTES", proposed_sidecar_page),
        ("SIDECAR_ENTRY_MAX_RAW_BYTES", resolved.sidecar_entry_max_raw_bytes),
        ("SIDECAR_ENTRY_MAX_BYTES", proposed_entry),
        ("SIDECAR_AGGREGATE_MAX_BYTES",
         _ceil_to(max(sidecar_aggregate, 1_048_576), 1_048_576)),
        ("ROOT_MAX_BYTES", _ceil_to(
            max(root_bytes, refused_root_bytes, planless_refused_root_bytes,
                resolved.root_max_bytes), 4_096)),
        ("UNIVERSE_MAX_ROWS", cap),
    )
    measured = (
        ("index_row_worst_case_bytes", index_row_bytes),
        ("index_page_envelope_bytes", index_envelope),
        ("index_page_worst_case_bytes", index_page_bytes),
        ("digest_page_envelope_bytes", digest_envelope),
        ("digest_page_worst_case_bytes", digest_page_bytes),
        ("sidecar_entry_worst_case_bytes", entry_bytes),
        ("sidecar_page_envelope_bytes", sidecar_envelope),
        ("sidecar_page_worst_case_bytes", sidecar_page_bytes),
        ("sidecar_aggregate_worst_case_bytes", sidecar_aggregate),
        ("ingress_record_worst_case_bytes", ingress_record_bytes),
        ("ingress_tuple_worst_case_bytes", ingress_tuple_bytes),
        ("refused_root_worst_case_bytes", refused_root_bytes),
        ("planless_refused_root_worst_case_bytes", planless_refused_root_bytes),
        ("root_worst_case_bytes", root_bytes),
    )
    derived = (
        ("max_rows_per_index_page", max_index_rows or 0),
        ("index_page_count", index_page_count),
        ("max_digests_per_digest_page", max_digests or 0),
        ("digest_page_count_per_section", 0 if not max_digests else (
            (cap + max_digests - 1) // max_digests)),
        ("digest_page_count", digest_page_count),
        ("max_entries_per_sidecar_page", max_entries or 0),
        ("sidecar_page_count", sidecar_page_count),
    )
    return CompleteSupportPreflightV1(
        preflight_status=status, preflight_refusal_reason=reason,
        declared_universe_cap=cap, cap_invariant_ok=invariant_ok,
        measured_bytes=measured, derived_counts=derived, proposed_caps=proposed,
        detail=("" if not over else "exceeded=" + ",".join(over)))
