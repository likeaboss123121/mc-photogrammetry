"""Evaluation-only candidate-support provenance and Phase-B ownership projection.

This module is the single additive, default-off seam for
``DS-IMG-CANDIDATE-SUPPORT-PROVENANCE-IMPL-R4``. It owns four things and nothing
else:

* :class:`FrameIngressProvenanceV1` - the typed ingress pin (encoded-byte
  SHA-256/length, decoded shape/dtype/pixel digest, reversible
  :class:`FrameTransform`, decoder provenance) captured from the one shared
  encoded-byte read that feeds both discovery and decode;
* :class:`SourceTapWitnessBindingV1` - the ordinary ``plane_signature`` call's
  own :class:`SignatureSamplingWitnessV1` bound to that ingress pin, the usable
  mask, the signature digest and the sampler settings;
* :class:`CandidateSupportProvenanceV1` - the strict, bounded, fail-closed
  record over the complete pre-filter admitted universe (kept and dropped), and
  :func:`failed_candidate_support_provenance`, its collector;
* :func:`project_ownership_into_source_space` - the pure Phase-B join that maps
  the frozen source-pixel indices into a separately pinned, synthetic evaluator
  ownership map **projected into exactly that source space**. It never replays
  ``plane_signature`` on evaluator pixels.

Nothing here runs on the default production path. No evaluator truth, camera
manifest, mask payload, Minecraft or JAR input is read, and no real screenshot
fixture is opened.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from mc_photogrammetry.pipeline.contracts import FrameTransform
from mc_photogrammetry.pipeline.frame_ingress import (
    BYTE_DIGEST_MISMATCH,
    FRAME_TRANSFORM_UNAVAILABLE,
    INGRESS_DECODER_PROVENANCE,
    INGRESS_PROVENANCE_SCHEMA,
    INGRESS_PROVENANCE_UNAVAILABLE,
    FrameIngressProvenanceV1,
    build_frame_ingress_provenance,
)
from mc_photogrammetry.reconstruction import lattice_planes
from mc_photogrammetry.reconstruction import plane_graph

__all__ = [
    "INGRESS_PROVENANCE_SCHEMA",
    "INGRESS_DECODER_PROVENANCE",
    "INGRESS_PROVENANCE_UNAVAILABLE",
    "BYTE_DIGEST_MISMATCH",
    "FRAME_TRANSFORM_UNAVAILABLE",
    "SOURCE_TAP_WITNESS_UNAVAILABLE",
    "SOURCE_TAP_WITNESS_BUDGET_EXHAUSTED",
    "SOURCE_TAP_WITNESS_SHAPE_MISMATCH",
    "SOURCE_TAP_WITNESS_INDEX_OUT_OF_RANGE",
    "SUPPORT_PAYLOAD_UNAVAILABLE",
    "SUPPORT_PAYLOAD_OVER_CAP",
    "CANDIDATE_SUPPORT_SCHEMA",
    "CANDIDATE_SUPPORT_BASIS",
    "CANDIDATE_SUPPORT_DIGEST_DOMAIN",
    "INPUT_MATCH_IDENTITY_DOMAIN",
    "INPUT_MATCH_IDENTITY_SCHEMA",
    "CANDIDATE_SUPPORT_MAX_ROWS",
    "CANDIDATE_SUPPORT_MAX_RECORD_BYTES",
    "CANDIDATE_SUPPORT_MAX_SUPPORT_BYTES",
    "CANDIDATE_SUPPORT_ORIENTATION",
    "CandidateSupportError",
    "build_frame_ingress_provenance",
    "FrameIngressProvenanceV1",
    "SourceTapWitnessBindingV1",
    "SourceTapWitnessRefusalV1",
    "MatcherSupportPayloadV1",
    "CandidateSupportRowV1",
    "CandidateSupportProvenanceV1",
    "validate_candidate_support_provenance",
    "candidate_support_record_digest",
    "failed_candidate_support_provenance",
    "ProjectedEvaluatorOwnershipV1",
    "OwnershipProjectionV1",
    "OwnershipProjectionAbstentionV1",
    "project_ownership_into_source_space",
]

SOURCE_TAP_WITNESS_UNAVAILABLE = "source_tap_witness_unavailable"
SOURCE_TAP_WITNESS_BUDGET_EXHAUSTED = "source_tap_witness_budget_exhausted"
SOURCE_TAP_WITNESS_SHAPE_MISMATCH = "source_tap_witness_shape_mismatch"
SOURCE_TAP_WITNESS_INDEX_OUT_OF_RANGE = "source_tap_witness_index_out_of_range"
SUPPORT_PAYLOAD_UNAVAILABLE = "support_payload_unavailable"
SUPPORT_PAYLOAD_OVER_CAP = "support_payload_over_cap"
MATCHER_WITNESS_UNAVAILABLE = "matcher_witness_unavailable"

CANDIDATE_SUPPORT_SCHEMA = "mc-photogrammetry-candidate-support-provenance-v1"
CANDIDATE_SUPPORT_BASIS = "derived_from_pre_filter_admitted_universe"
CANDIDATE_SUPPORT_DIGEST_DOMAIN = "candidate_support_provenance"
CANDIDATE_SUPPORT_ORIENTATION = "left-source-right-target"

# The shared post-filter solver-input identity domain/schema used by the R7
# ``SolverInputBindings`` and the R9 ``SameAxisCandidateCapture`` records. The
# candidate-support record binds the exact same digest so the three records can
# be shown to describe the same ordered match sequence.
INPUT_MATCH_IDENTITY_DOMAIN = "input_match_identity"
INPUT_MATCH_IDENTITY_SCHEMA = "mc-photogrammetry-failed-active-edge-aliases-v1"

CANDIDATE_SUPPORT_MAX_ROWS = 1_024
CANDIDATE_SUPPORT_MAX_RECORD_BYTES = 262_144
CANDIDATE_SUPPORT_MAX_SUPPORT_BYTES = 131_072
CANDIDATE_SUPPORT_TOLERANCE = 0.15


class CandidateSupportError(ValueError):
    """A typed, bounded refusal raised by this module's builders."""

    def __init__(self, reason: str, field: str = "", detail: str = "") -> None:
        self.reason = reason
        self.field = field
        self.detail = detail
        super().__init__(reason if not detail else f"{reason}: {detail}")


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")).hexdigest()


def _hex64(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


@dataclass(frozen=True, slots=True)
class SourceTapWitnessRefusalV1:
    """Typed refusal to bind one accepted signature to its source-tap witness."""

    frame_id: str
    canonical_index: int
    alternative_index: int
    reason: str
    detail: str = ""

    def __post_init__(self) -> None:
        if (type(self.frame_id) is not str or not self.frame_id
                or type(self.canonical_index) is not int or self.canonical_index < 0
                or type(self.alternative_index) is not int
                or self.alternative_index < 0
                or self.reason not in (
                    INGRESS_PROVENANCE_UNAVAILABLE, SOURCE_TAP_WITNESS_UNAVAILABLE,
                    SOURCE_TAP_WITNESS_BUDGET_EXHAUSTED,
                    SOURCE_TAP_WITNESS_SHAPE_MISMATCH,
                    SOURCE_TAP_WITNESS_INDEX_OUT_OF_RANGE)
                or type(self.detail) is not str):
            raise CandidateSupportError(SOURCE_TAP_WITNESS_UNAVAILABLE, "refusal")


@dataclass(frozen=True, slots=True)
class SourceTapWitnessBindingV1:
    """One accepted signature bound to the witness its own sampler call emitted."""

    frame_id: str
    canonical_index: int
    alternative_index: int
    signature_sha256: str
    ingress: FrameIngressProvenanceV1
    mask_sha256: str
    masks_sha256: str
    max_cells: int
    max_rectified_side: int
    pixels_per_cell: int
    witness: lattice_planes.SignatureSamplingWitnessV1
    witness_sha256: str = ""
    schema: str = "mc-photogrammetry-source-tap-witness-binding-v1"

    def __post_init__(self) -> None:
        if (self.schema != "mc-photogrammetry-source-tap-witness-binding-v1"
                or type(self.frame_id) is not str or not self.frame_id
                or type(self.canonical_index) is not int or self.canonical_index < 0
                or type(self.alternative_index) is not int or self.alternative_index < 0
                or not _hex64(self.signature_sha256)
                or not isinstance(self.ingress, FrameIngressProvenanceV1)
                or self.ingress.frame_id != self.frame_id
                or not _hex64(self.mask_sha256) or not _hex64(self.masks_sha256)
                or not _positive_int(self.max_cells)
                or not _positive_int(self.max_rectified_side)
                or not _positive_int(self.pixels_per_cell)
                or not isinstance(self.witness, lattice_planes.SignatureSamplingWitnessV1)):
            raise CandidateSupportError(SOURCE_TAP_WITNESS_UNAVAILABLE, "binding")
        shape = self.witness.rgb_rectification.source_shape
        if shape != self.ingress.source_shape:
            raise CandidateSupportError(SOURCE_TAP_WITNESS_SHAPE_MISMATCH, "source_shape")
        for rectification in (self.witness.rgb_rectification,
                              self.witness.coverage_rectification,
                              self.witness.region_rectification):
            if rectification is None:
                continue
            if rectification.source_shape != shape:
                raise CandidateSupportError(SOURCE_TAP_WITNESS_SHAPE_MISMATCH, "source_shape")
            # ``-1`` is the rectification tap table's documented border sentinel
            # for a tap that falls outside the decoded source frame (see
            # ``lattice.rectification_witness_v1``); the sibling
            # ``cell_source_indices`` check below accepts exactly the same
            # sentinel. ``0`` is a real pixel index, and anything below ``-1`` or
            # at/above ``rows * columns`` fails closed.
            if np.any((rectification.tap_source_indices < -1)
                      | (rectification.tap_source_indices >= shape[0] * shape[1])):
                raise CandidateSupportError(
                    SOURCE_TAP_WITNESS_INDEX_OUT_OF_RANGE, "tap_source_indices")
        indices = self.witness.cell_source_indices
        if np.any(indices < -1) or np.any(indices >= shape[0] * shape[1]):
            raise CandidateSupportError(
                SOURCE_TAP_WITNESS_INDEX_OUT_OF_RANGE, "cell_source_indices")
        if self.witness_sha256 not in ("", self.witness.witness_sha256):
            raise CandidateSupportError(SOURCE_TAP_WITNESS_UNAVAILABLE, "witness_sha256")
        object.__setattr__(self, "witness_sha256", self.witness.witness_sha256)

    def to_record(self) -> dict:
        return {
            "schema": self.schema, "frame_id": self.frame_id,
            "canonical_index": self.canonical_index,
            "alternative_index": self.alternative_index,
            "signature_sha256": self.signature_sha256,
            "ingress_sha256": _sha(self.ingress.to_record()),
            "mask_sha256": self.mask_sha256, "masks_sha256": self.masks_sha256,
            "max_cells": self.max_cells,
            "max_rectified_side": self.max_rectified_side,
            "pixels_per_cell": self.pixels_per_cell,
            "witness_sha256": self.witness_sha256,
        }


@dataclass(frozen=True, slots=True)
class MatcherSupportPayloadV1:
    """Retrievable, bounded matcher-support payload for one admitted row."""

    logical_pairs: np.ndarray
    warp_source_rows: np.ndarray
    warp_source_columns: np.ndarray
    source_shape: tuple[int, int]
    target_shape: tuple[int, int]
    payload_sha256: str = ""
    schema: str = "mc-photogrammetry-matcher-support-payload-v1"

    def __post_init__(self) -> None:
        if (self.schema != "mc-photogrammetry-matcher-support-payload-v1"
                or not isinstance(self.logical_pairs, np.ndarray)
                or self.logical_pairs.ndim != 2 or self.logical_pairs.shape[1] != 4
                or self.logical_pairs.dtype != np.int64
                or self.logical_pairs.flags.writeable
                or not isinstance(self.warp_source_rows, np.ndarray)
                or not isinstance(self.warp_source_columns, np.ndarray)
                or self.warp_source_rows.dtype != np.int64
                or self.warp_source_columns.dtype != np.int64
                or self.warp_source_rows.flags.writeable
                or self.warp_source_columns.flags.writeable
                or self.warp_source_rows.shape != self.target_shape
                or self.warp_source_columns.shape != self.target_shape):
            raise CandidateSupportError(SUPPORT_PAYLOAD_UNAVAILABLE, "payload")
        if (len(self.source_shape) != 2 or len(self.target_shape) != 2
                or any(not _positive_int(value) for value in self.source_shape + self.target_shape)):
            raise CandidateSupportError(SUPPORT_PAYLOAD_UNAVAILABLE, "shapes")
        # C5: every selected logical pair is a real source/target cell. The ``-1``
        # border sentinel is allowed only in the warp sampling map
        # (``warp_source_rows``/``warp_source_columns``), never in a selected pair,
        # and ``0`` is a real index rather than a sentinel.
        if self.logical_pairs.size:
            pairs = self.logical_pairs
            if int(pairs.min()) < 0:
                raise CandidateSupportError(SUPPORT_PAYLOAD_UNAVAILABLE, "logical_pairs")
            if (int(pairs[:, 0].max()) >= self.source_shape[0]
                    or int(pairs[:, 1].max()) >= self.source_shape[1]
                    or int(pairs[:, 2].max()) >= self.target_shape[0]
                    or int(pairs[:, 3].max()) >= self.target_shape[1]):
                raise CandidateSupportError(SUPPORT_PAYLOAD_UNAVAILABLE, "logical_pairs")
        digest = _sha(self._payload(include_digest=False))
        if self.payload_sha256 not in ("", digest):
            raise CandidateSupportError(SUPPORT_PAYLOAD_UNAVAILABLE, "payload_sha256")
        object.__setattr__(self, "payload_sha256", digest)

    def _payload(self, *, include_digest: bool) -> dict:
        def row(value: np.ndarray) -> dict:
            return {"dtype": value.dtype.str, "shape": list(value.shape),
                    "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
                    "values": value.tolist()}
        result = {
            "schema": self.schema,
            "source_shape": list(self.source_shape),
            "target_shape": list(self.target_shape),
            "logical_pairs": row(self.logical_pairs),
            "warp_source_rows": row(self.warp_source_rows),
            "warp_source_columns": row(self.warp_source_columns),
        }
        if include_digest:
            result["payload_sha256"] = self.payload_sha256
        return result

    def to_record(self, *, include_arrays: bool = True) -> dict:
        payload = self._payload(include_digest=True)
        if not include_arrays:
            for key in ("logical_pairs", "warp_source_rows", "warp_source_columns"):
                payload[key].pop("values", None)
        return payload

    def payload_bytes(self) -> int:
        return sum(value.nbytes for value in (
            self.logical_pairs, self.warp_source_rows, self.warp_source_columns))


@dataclass(frozen=True, slots=True)
class CandidateSupportRowV1:
    """One pre-filter admitted candidate, kept or dropped, with its own support."""

    admitted_row_index: int
    group_index: int
    left: str
    right: str
    left_axis: int
    right_axis: int
    quadrant_of_right: int
    disposition: str
    delta: float
    tolerance: float
    match_index: int | None
    left_plane: int
    right_plane: int
    left_alternative_index: int
    right_alternative_index: int
    left_b: float
    right_b: float
    input_match_identity: tuple
    input_match_identity_sha256: str
    match_sha256: str
    envelope_sha256: str | None
    left_signature_sha256: str | None
    right_signature_sha256: str | None
    matcher_witness_sha256: str | None
    matcher_refusal_reason: str | None
    overlap_cells: int
    cell_shift_rc: tuple[int, int] | None
    support: MatcherSupportPayloadV1 | None
    support_refusal_reason: str | None
    orientation: str = CANDIDATE_SUPPORT_ORIENTATION

    def __post_init__(self) -> None:
        if (type(self.admitted_row_index) is not int or self.admitted_row_index < 0
                or type(self.group_index) is not int or self.group_index < 0
                or type(self.left) is not str or not self.left
                or type(self.right) is not str or not self.right
                or type(self.left_axis) is not int or type(self.right_axis) is not int
                or type(self.quadrant_of_right) is not int
                or self.disposition not in (
                    plane_graph.SAME_AXIS_DISPOSITION_KEPT,
                    plane_graph.SAME_AXIS_DISPOSITION_DROPPED)
                or type(self.left_plane) is not int or type(self.right_plane) is not int
                or type(self.left_alternative_index) is not int
                or self.left_alternative_index < 0
                or type(self.right_alternative_index) is not int
                or self.right_alternative_index < 0
                or self.orientation != CANDIDATE_SUPPORT_ORIENTATION
                or not _hex64(self.input_match_identity_sha256)
                or not _hex64(self.match_sha256)
                or (self.envelope_sha256 is not None and not _hex64(self.envelope_sha256))
                or type(self.overlap_cells) is not int or self.overlap_cells < 0):
            raise ValueError("candidate support row fields are malformed")
        if self.disposition == plane_graph.SAME_AXIS_DISPOSITION_KEPT:
            if type(self.match_index) is not int or self.match_index < 0:
                raise ValueError("kept candidate support row needs a match index")
        elif self.match_index is not None:
            raise ValueError("dropped candidate support row must not claim a match index")
        if (self.support is None) == (self.support_refusal_reason is None):
            raise ValueError("candidate support row needs exactly one support outcome")
        if self.support is not None:
            if len(self.support.logical_pairs) != self.overlap_cells:
                raise ValueError("candidate support workload does not conserve pairs")
        if self.matcher_refusal_reason is not None and self.support is not None:
            raise ValueError("a refused matcher witness cannot carry a support payload")

    def to_record(self) -> dict:
        return {
            "admitted_row_index": self.admitted_row_index,
            "group_index": self.group_index,
            "left": self.left, "right": self.right,
            "left_axis": self.left_axis, "right_axis": self.right_axis,
            "quadrant_of_right": self.quadrant_of_right,
            "disposition": self.disposition, "delta": float(self.delta),
            "tolerance": float(self.tolerance), "match_index": self.match_index,
            "left_plane": self.left_plane, "right_plane": self.right_plane,
            "left_alternative_index": self.left_alternative_index,
            "right_alternative_index": self.right_alternative_index,
            "left_b": float(self.left_b), "right_b": float(self.right_b),
            "input_match_identity": list(self.input_match_identity),
            "input_match_identity_sha256": self.input_match_identity_sha256,
            "match_sha256": self.match_sha256,
            "envelope_sha256": self.envelope_sha256,
            "left_signature_sha256": self.left_signature_sha256,
            "right_signature_sha256": self.right_signature_sha256,
            "matcher_witness_sha256": self.matcher_witness_sha256,
            "matcher_refusal_reason": self.matcher_refusal_reason,
            "overlap_cells": self.overlap_cells,
            "cell_shift_rc": (None if self.cell_shift_rc is None
                              else list(self.cell_shift_rc)),
            "orientation": self.orientation,
            "support": None if self.support is None else self.support.to_record(),
            "support_refusal_reason": self.support_refusal_reason,
        }


@dataclass(frozen=True, slots=True)
class CandidateSupportProvenanceV1:
    """Strict bounded record over the complete pre-filter admitted universe."""

    status: str
    basis: str | None
    reason: str | None
    field: str | None
    failed_assignment_sha256: str
    input_match_identity_sha256: str
    row_count: int
    ingress: tuple[FrameIngressProvenanceV1, ...]
    rows: tuple[CandidateSupportRowV1, ...]
    schema: str = CANDIDATE_SUPPORT_SCHEMA
    record_sha256: str = ""

    def __post_init__(self) -> None:
        # Compute and bind the canonical digest before strict validation so the
        # validator can replay ``record_sha256`` even during construction.
        digest = candidate_support_record_digest(self)
        if self.record_sha256 not in ("", digest):
            raise ValueError("candidate support record digest mismatch")
        object.__setattr__(self, "record_sha256", digest)
        validate_candidate_support_provenance(self)

    def to_record(self) -> dict:
        payload = candidate_support_payload(self)
        payload["record_sha256"] = self.record_sha256
        return payload


def candidate_support_payload(value: CandidateSupportProvenanceV1) -> dict:
    if value.status == "refused":
        return {"schema": value.schema, "status": "refused",
                "reason": value.reason, "field": value.field}
    return {
        "schema": value.schema, "status": "assembled", "basis": value.basis,
        "failed_assignment_sha256": value.failed_assignment_sha256,
        "input_match_identity_sha256": value.input_match_identity_sha256,
        "row_count": value.row_count,
        "ingress": [record.to_record() for record in value.ingress],
        "rows": [row.to_record() for row in value.rows],
    }


def candidate_support_record_digest(value: CandidateSupportProvenanceV1) -> str:
    return _sha({"domain": CANDIDATE_SUPPORT_DIGEST_DOMAIN, "schema": value.schema,
                 "payload": candidate_support_payload(value)})


def _refused(reason: str, field: str) -> CandidateSupportProvenanceV1:
    return CandidateSupportProvenanceV1(
        status="refused", basis=None, reason=reason, field=field,
        failed_assignment_sha256="", input_match_identity_sha256="",
        row_count=0, ingress=(), rows=())


def validate_candidate_support_provenance(value: object) -> None:
    """Strict, fail-closed validation of one candidate-support record."""
    if not isinstance(value, CandidateSupportProvenanceV1):
        raise ValueError("candidate support record has an invalid type")
    if value.schema != CANDIDATE_SUPPORT_SCHEMA:
        raise ValueError("candidate support record schema mismatch")
    if value.status == "refused":
        if (value.basis is not None or value.rows or value.ingress
                or value.row_count != 0 or not value.reason
                or type(value.field) is not str):
            raise ValueError("refused candidate support record must be atomic")
        return
    if value.status != "assembled" or value.reason is not None or value.field is not None:
        raise ValueError("candidate support record status is malformed")
    if value.basis != CANDIDATE_SUPPORT_BASIS:
        raise ValueError("candidate support record basis is malformed")
    if len(value.rows) != value.row_count or value.row_count == 0:
        raise ValueError("candidate support record row count is inconsistent")
    if value.row_count > CANDIDATE_SUPPORT_MAX_ROWS:
        raise ValueError("candidate support record exceeds the row cap")
    if (not isinstance(value.ingress, tuple)
            or any(not isinstance(item, FrameIngressProvenanceV1)
                   for item in value.ingress)):
        raise ValueError("candidate support ingress tuple is malformed")
    ingress_by_frame = {item.frame_id: item for item in value.ingress}
    if len(ingress_by_frame) != len(value.ingress):
        raise ValueError("candidate support ingress repeats a frame")
    seen: set[int] = set()
    joins: set[tuple] = set()
    for position, row in enumerate(value.rows):
        if not isinstance(row, CandidateSupportRowV1):
            raise ValueError("candidate support record holds a non-row value")
        if row.admitted_row_index in seen:
            raise ValueError("candidate support row index is duplicated")
        seen.add(row.admitted_row_index)
        if row.admitted_row_index != position:
            raise ValueError("candidate support rows are not in admitted order")
        join = (row.admitted_row_index, row.match_index, row.match_sha256,
                row.envelope_sha256, row.left_signature_sha256,
                row.right_signature_sha256, row.left_alternative_index,
                row.right_alternative_index)
        if join in joins:
            raise ValueError("candidate support join key is not one-to-one")
        joins.add(join)
        for frame_id in (row.left, row.right):
            if frame_id not in ingress_by_frame:
                raise ValueError("candidate support row names a frame with no ingress")
        if row.support is not None:
            support = row.support
            if len(support.logical_pairs) != row.overlap_cells:
                raise ValueError("candidate support pair count does not conserve")
            if support.logical_pairs.size:
                pairs = support.logical_pairs
                # C5: ``-1`` is the warp sampling map's border sentinel and is
                # never a selected logical pair; ``0`` is a real cell index. Any
                # negative or out-of-grid pair fails closed.
                if int(pairs.min()) < 0:
                    raise ValueError("candidate support selects a sentinel logical pair")
                if (int(pairs[:, 0].max()) >= support.source_shape[0]
                        or int(pairs[:, 1].max()) >= support.source_shape[1]
                        or int(pairs[:, 2].max()) >= support.target_shape[0]
                        or int(pairs[:, 3].max()) >= support.target_shape[1]):
                    raise ValueError("candidate support pair index is outside its grid")
            if support.payload_bytes() > CANDIDATE_SUPPORT_MAX_SUPPORT_BYTES:
                raise ValueError("candidate support payload exceeds the support cap")
    if value.input_match_identity_sha256 != _identity_digest(
            [list(row.input_match_identity) for row in value.rows
             if row.disposition == plane_graph.SAME_AXIS_DISPOSITION_KEPT]):
        raise ValueError("candidate support input identity digest does not replay")
    if not _hex64(value.failed_assignment_sha256):
        raise ValueError("candidate support failed assignment digest is malformed")
    if not _hex64(value.record_sha256):
        raise ValueError("candidate support record digest is malformed")
    if value.record_sha256 != candidate_support_record_digest(value):
        raise ValueError("candidate support record digest does not replay")


def _identity_digest(identity_rows: Sequence[list]) -> str:
    return _sha({"domain": INPUT_MATCH_IDENTITY_DOMAIN,
                 "schema": INPUT_MATCH_IDENTITY_SCHEMA,
                 "payload": [list(row) for row in identity_rows]})


def failed_candidate_support_provenance(
    readings: object,
    matches: object,
    signature_witnesses: object,
    matcher_witness_rows: object,
    same_axis_observations: object,
    *,
    ingress: Sequence[FrameIngressProvenanceV1] = (),
    failed_assignment_sha256: str = "",
    max_rows: int = CANDIDATE_SUPPORT_MAX_ROWS,
    max_record_bytes: int = CANDIDATE_SUPPORT_MAX_RECORD_BYTES,
    max_support_bytes: int = CANDIDATE_SUPPORT_MAX_SUPPORT_BYTES,
) -> CandidateSupportProvenanceV1:
    """Collect the strict, bounded candidate-support record or refuse atomically.

    ``matcher_witness_rows`` is the full-universe publication from
    :func:`plane_graph.match_plane_graph` (one row per sorted pre-filter admitted
    candidate, kept or dropped); ``same_axis_observations`` is the parallel
    :class:`plane_graph.SameAxisCandidateObservation`; ``signature_witnesses`` are
    the :class:`SourceTapWitnessBindingV1` values captured in the ordinary
    signature call. Nothing here re-matches, re-signatures, re-solves or opens a
    frame.
    """
    try:
        return _collect_candidate_support(
            readings, matches, signature_witnesses, matcher_witness_rows,
            same_axis_observations, ingress=tuple(ingress),
            failed_assignment_sha256=failed_assignment_sha256,
            max_rows=max_rows, max_record_bytes=max_record_bytes,
            max_support_bytes=max_support_bytes)
    except CandidateSupportError as error:
        return _refused(error.reason, error.field or "record")
    except (AttributeError, TypeError, ValueError, IndexError, KeyError,
            OverflowError, ZeroDivisionError) as error:
        return _refused(SUPPORT_PAYLOAD_UNAVAILABLE, type(error).__name__)


def _collect_candidate_support(
    readings: object,
    matches: object,
    signature_witnesses: object,
    matcher_witness_rows: object,
    same_axis_observations: object,
    *,
    ingress: tuple[FrameIngressProvenanceV1, ...],
    failed_assignment_sha256: str,
    max_rows: int,
    max_record_bytes: int,
    max_support_bytes: int,
) -> CandidateSupportProvenanceV1:
    if not isinstance(same_axis_observations, plane_graph.SameAxisCandidateObservation):
        raise CandidateSupportError(SUPPORT_PAYLOAD_UNAVAILABLE, "same_axis_observation")
    observations = same_axis_observations
    if not isinstance(matcher_witness_rows, (list, tuple)):
        raise CandidateSupportError(SUPPORT_PAYLOAD_UNAVAILABLE, "matcher_witness_rows")
    if len(matcher_witness_rows) != len(observations.candidates):
        raise CandidateSupportError(SUPPORT_PAYLOAD_UNAVAILABLE, "matcher_witness_rows")
    if len(observations.candidates) > max_rows:
        raise CandidateSupportError("budget_exhausted", "row_count")
    if not isinstance(signature_witnesses, (list, tuple)):
        raise CandidateSupportError(SOURCE_TAP_WITNESS_UNAVAILABLE, "signature_witnesses")
    witness_by_signature: dict[str, SourceTapWitnessBindingV1] = {}
    for item in signature_witnesses:
        if isinstance(item, SourceTapWitnessBindingV1):
            witness_by_signature[item.signature_sha256] = item

    kept_positions = {id(match): index for index, match in enumerate(matches)}
    groups: dict[tuple, int] = {}
    rows: list[CandidateSupportRowV1] = []
    identity_rows: list[list] = []
    for position, (candidate, disposition) in enumerate(zip(
            observations.candidates, observations.dispositions)):
        row = matcher_witness_rows[position]
        if not isinstance(row, plane_graph.PreFilterMatcherWitnessRowV1):
            raise CandidateSupportError(SUPPORT_PAYLOAD_UNAVAILABLE, "matcher_witness_row")
        if (row.admitted_row_index != position or row.disposition != disposition):
            raise CandidateSupportError(
                SUPPORT_PAYLOAD_UNAVAILABLE, "row_disposition_binding")
        group_key = (candidate.left, candidate.right, candidate.left_axis,
                     candidate.quadrant_of_right)
        group_index = groups.setdefault(group_key, len(groups))
        identity = [candidate.left, candidate.left_plane, candidate.left_axis,
                    candidate.right, candidate.right_plane, candidate.right_axis,
                    candidate.quadrant_of_right, float(candidate.left_b),
                    float(candidate.right_b)]
        match_index = (kept_positions.get(id(candidate))
                       if disposition == plane_graph.SAME_AXIS_DISPOSITION_KEPT else None)
        left_witness = witness_by_signature.get(row.left_signature_sha256 or "")
        right_witness = witness_by_signature.get(row.right_signature_sha256 or "")
        left_alt = _alternative_index(readings, candidate.left, candidate.left_plane,
                                      candidate.left_b)
        right_alt = _alternative_index(readings, candidate.right, candidate.right_plane,
                                       candidate.right_b)
        support = _support_payload(row)
        if support is not None and support.payload_bytes() > max_support_bytes:
            raise CandidateSupportError(SUPPORT_PAYLOAD_OVER_CAP, "support_bytes")
        if support is not None:
            if left_witness is None or right_witness is None:
                raise CandidateSupportError(
                    SOURCE_TAP_WITNESS_UNAVAILABLE, "signature_witness")
            left_shape = left_witness.ingress.source_shape
            right_shape = right_witness.ingress.source_shape
            if left_shape != right_shape:
                raise CandidateSupportError(
                    SOURCE_TAP_WITNESS_SHAPE_MISMATCH, "signature_witness")
        rows.append(CandidateSupportRowV1(
            admitted_row_index=position, group_index=group_index,
            left=candidate.left, right=candidate.right,
            left_axis=int(candidate.left_axis), right_axis=int(candidate.right_axis),
            quadrant_of_right=int(candidate.quadrant_of_right),
            disposition=disposition, delta=_delta(candidate, readings),
            tolerance=CANDIDATE_SUPPORT_TOLERANCE, match_index=match_index,
            left_plane=int(candidate.left_plane), right_plane=int(candidate.right_plane),
            left_alternative_index=left_alt, right_alternative_index=right_alt,
            left_b=float(candidate.left_b), right_b=float(candidate.right_b),
            input_match_identity=tuple(identity),
            input_match_identity_sha256=_identity_digest([identity]),
            match_sha256=row.match_sha256,
            envelope_sha256=row.envelope_sha256,
            left_signature_sha256=row.left_signature_sha256 or None,
            right_signature_sha256=row.right_signature_sha256 or None,
            matcher_witness_sha256=(None if row.payload is None
                                    else row.payload.witness.witness_sha256),
            matcher_refusal_reason=row.refusal_reason,
            overlap_cells=int(candidate.overlap_cells),
            cell_shift_rc=(None if candidate.cell_shift_rc is None
                           else (int(candidate.cell_shift_rc[0]),
                                 int(candidate.cell_shift_rc[1]))),
            support=support,
            support_refusal_reason=(None if support is not None
                                    else (row.refusal_reason or SUPPORT_PAYLOAD_UNAVAILABLE))))
        if disposition == plane_graph.SAME_AXIS_DISPOSITION_KEPT:
            identity_rows.append(identity)
    if not rows:
        raise CandidateSupportError(SUPPORT_PAYLOAD_UNAVAILABLE, "candidates")
    value = CandidateSupportProvenanceV1(
        status="assembled", basis=CANDIDATE_SUPPORT_BASIS, reason=None, field=None,
        failed_assignment_sha256=failed_assignment_sha256 or _sha(
            {"domain": "failed_best_assignment", "payload": []}),
        input_match_identity_sha256=_identity_digest(identity_rows),
        row_count=len(rows), ingress=ingress, rows=tuple(rows))
    encoded = json.dumps(candidate_support_payload(value), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=True,
                         allow_nan=False).encode("utf-8")
    if len(encoded) > max_record_bytes:
        raise CandidateSupportError("budget_exhausted", "record_bytes")
    return value


def _alternative_index(readings: object, frame: str, plane: int, b: float) -> int:
    index, _by_object, _frames = plane_graph._canonical_reading_index(readings)
    reading = index.get((frame, plane))
    if reading is None:
        raise CandidateSupportError(SUPPORT_PAYLOAD_UNAVAILABLE, "reading_index")
    for alternative_index, alternative in enumerate(reading.alternatives):
        if float(alternative.blocks_per_period) == float(b):
            return alternative_index
    raise CandidateSupportError(SUPPORT_PAYLOAD_UNAVAILABLE, "alternative_index")


def _delta(candidate: plane_graph.PlaneMatch, readings: object) -> float:
    index, _by_object, _frames = plane_graph._canonical_reading_index(readings)
    value = plane_graph._delta(candidate, index)
    if value is None:
        raise CandidateSupportError(SUPPORT_PAYLOAD_UNAVAILABLE, "same_axis_delta")
    return float(value)


def _support_payload(row: plane_graph.PreFilterMatcherWitnessRowV1) -> MatcherSupportPayloadV1 | None:
    if row.payload is None:
        return None
    envelope = row.payload.envelope
    witness = row.payload.witness
    warp = witness.warp_witness
    if warp is None:
        raise CandidateSupportError(SUPPORT_PAYLOAD_UNAVAILABLE, "warp_witness")
    logical_pairs = np.asarray(witness.logical_pairs, dtype=np.int64)
    logical_pairs = np.frombuffer(logical_pairs.tobytes(order="C"),
                                  dtype=np.int64).reshape(logical_pairs.shape)
    rows = np.frombuffer(np.ascontiguousarray(warp.source_rows, dtype=np.int64).tobytes(
        order="C"), dtype=np.int64).reshape(warp.source_rows.shape)
    columns = np.frombuffer(np.ascontiguousarray(
        warp.source_columns, dtype=np.int64).tobytes(order="C"),
        dtype=np.int64).reshape(warp.source_columns.shape)
    if len(logical_pairs) != envelope.overlap_cells:
        raise CandidateSupportError(SUPPORT_PAYLOAD_UNAVAILABLE, "logical_pairs")
    return MatcherSupportPayloadV1(
        logical_pairs=logical_pairs, warp_source_rows=rows,
        warp_source_columns=columns, source_shape=warp.source_shape,
        target_shape=warp.target_shape)


# --- Phase B: project evaluator ownership into the frozen source space --------

@dataclass(frozen=True, slots=True)
class ProjectedEvaluatorOwnershipV1:
    """A separately pinned, synthetic evaluator ownership map.

    ``ownership`` is the per-pixel integer owner id (``-1`` background) in the
    **frozen source coordinate space**; ``usable`` is the pinned evaluator
    usability convention. ``owner_plane_keys`` maps an owner id to its physical
    plane key (or ``None`` when unresolved). This is evaluation-only material and
    is never written into a Phase A record.
    """

    ownership: np.ndarray
    usable: np.ndarray
    frame_size: tuple[int, int]
    camera_sha256: str
    transform: FrameTransform
    occlusion_convention: str
    owner_plane_keys: Mapping[int, object] = field(default_factory=dict)
    schema: str = "mc-photogrammetry-projected-evaluator-ownership-v1"


@dataclass(frozen=True, slots=True)
class OwnershipProjectionV1:
    cell_states: tuple[str, ...]
    cell_plane_keys: tuple[object, ...]
    reason_counts: tuple[tuple[str, int], ...]
    schema: str = "mc-photogrammetry-ownership-projection-v1"


@dataclass(frozen=True, slots=True)
class OwnershipProjectionAbstentionV1:
    reason: str
    field: str = ""
    schema: str = "mc-photogrammetry-ownership-projection-abstention-v1"


_OWNERSHIP_REASONS = (
    "absent_owner_records", "empty_dependencies", "out_of_range_indices",
    "unresolved_physical_plane_identity", "unusable_source_pixels")


def project_ownership_into_source_space(
    *,
    binding: SourceTapWitnessBindingV1,
    evaluator: ProjectedEvaluatorOwnershipV1,
    camera_sha256: str,
    inference_transform: FrameTransform,
    occlusion_convention: str,
) -> OwnershipProjectionV1 | OwnershipProjectionAbstentionV1:
    """Map frozen source-pixel indices into projected evaluator ownership.

    The join key is the saved witness index and its digests, never a re-sampled
    signature. No ``plane_signature`` call is made here, no evaluator RGB is read
    and no ownership result feeds back into inference.
    """
    if not isinstance(binding, SourceTapWitnessBindingV1):
        return OwnershipProjectionAbstentionV1("source_projection_unavailable", "binding")
    if not isinstance(evaluator, ProjectedEvaluatorOwnershipV1):
        return OwnershipProjectionAbstentionV1("source_projection_unavailable", "evaluator")
    ownership = np.asarray(evaluator.ownership)
    usable = np.asarray(evaluator.usable, dtype=bool)
    if ownership.ndim != 2 or usable.shape != ownership.shape:
        return OwnershipProjectionAbstentionV1("dimension_mismatch", "ownership")
    if evaluator.frame_size != (ownership.shape[1], ownership.shape[0]):
        return OwnershipProjectionAbstentionV1("dimension_mismatch", "frame_size")
    if evaluator.frame_size != binding.ingress.transform.resolved_output_size:
        return OwnershipProjectionAbstentionV1("dimension_mismatch", "transform")
    if evaluator.transform != inference_transform:
        return OwnershipProjectionAbstentionV1("transform_mismatch", "transform")
    if camera_sha256 != evaluator.camera_sha256:
        return OwnershipProjectionAbstentionV1(
            "evaluator_camera_misalignment", "camera")
    if occlusion_convention != evaluator.occlusion_convention:
        return OwnershipProjectionAbstentionV1("occlusion_unavailable", "occlusion")
    if binding.ingress.source_shape != ownership.shape:
        return OwnershipProjectionAbstentionV1("dimension_mismatch", "source_shape")
    flat_owners = ownership.reshape(-1)
    flat_usable = usable.reshape(-1)
    witness = binding.witness
    counts = {name: 0 for name in _OWNERSHIP_REASONS}
    states: list[str] = []
    keys: list[object] = []
    for cell in range(len(witness.cell_rectangles)):
        indices = witness.cell_source_indices[
            witness.cell_source_ptr[cell]:witness.cell_source_ptr[cell + 1]]
        causes: set[str] = set()
        resolved: set[object] = set()
        if len(indices) == 0:
            causes.add("empty_dependencies")
        for raw in indices:
            position = int(raw)
            if position < 0 or position >= flat_owners.size:
                causes.add("out_of_range_indices")
                continue
            owner_id = int(flat_owners[position])
            if not bool(flat_usable[position]):
                causes.add("unusable_source_pixels")
            key = evaluator.owner_plane_keys.get(owner_id)
            if key is None:
                causes.add("absent_owner_records")
            elif isinstance(key, tuple) and all(value is None for value in key):
                causes.add("unresolved_physical_plane_identity")
            else:
                resolved.add(key)
        for cause in causes:
            counts[cause] += 1
        if causes:
            states.append("unknown")
            keys.append(None)
        elif len(resolved) == 1:
            states.append("pure")
            keys.append(next(iter(resolved)))
        else:
            states.append("mixed")
            keys.append(None)
    return OwnershipProjectionV1(
        cell_states=tuple(states), cell_plane_keys=tuple(keys),
        reason_counts=tuple((name, counts[name]) for name in _OWNERSHIP_REASONS))
