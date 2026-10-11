"""Bounded, complete joint integer-shift domains for accepted C1 matches.

This module preserves every support-admissible correlation shift inside one
explicit radius. It does not rank those shifts or select a translation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np

from mc_photogrammetry.reconstruction import lattice_planes, plane_axis_support, plane_graph


SHIFT_DOMAIN_REFUSAL_REASONS = frozenset({
    "budget_exhausted",
    "selected_identity_missing",
    "selected_identity_ambiguous",
    "signature_missing",
    "signature_ambiguous",
    "signature_identity_mismatch",
    "signature_content_invalid",
    "homography_invalid",
    "provenance_mismatch",
    "correlation_replay_mismatch",
    "correlation_nonfinite",
    "shift_domain_incomplete",
})
SHIFT_DOMAIN_SCHEMA = "mc-photogrammetry-plane-shift-domain-v2"
SHIFT_DOMAIN_VERSION = 2
_DIGEST_LENGTH = 64
_MAX_DETAIL_LENGTH = 160


def _valid_digest(value: Any) -> bool:
    return (type(value) is str and len(value) == _DIGEST_LENGTH
            and all(char in "0123456789abcdef" for char in value))


def _strict_int(value: Any, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _json_value(value: Any) -> Any:
    """Return canonical JSON primitives, rejecting NaN and opaque objects."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite JSON value")
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise ValueError("JSON mapping keys must be strings")
        return {key: _json_value(value[key]) for key in sorted(value)}
    raise ValueError("value is not canonical JSON data")


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("ascii"))


def _float_record(value: Any) -> dict[str, str | float]:
    if type(value) not in (int, float):
        return {"kind": "invalid_type", "value": type(value).__name__}
    number = float(value)
    if math.isinf(number) and number > 0.0:
        return {"kind": "positive_infinity"}
    if math.isinf(number):
        return {"kind": "negative_infinity"}
    if math.isnan(number):
        return {"kind": "nan"}
    return {"kind": "finite", "value": number}


def _identity_key(identity: plane_axis_support.C1MatchIdentity) -> tuple:
    return identity.sort_key()


def _identity_record(identity: plane_axis_support.C1MatchIdentity) -> dict[str, Any]:
    return identity.to_record()


@dataclass(frozen=True, slots=True)
class ShiftDomainBudget:
    """Explicit whole-request limits; no resource limit is implicit."""

    max_domains: int
    max_padded_fft_cells: int
    max_shift_positions: int
    max_overlap_cell_visits: int
    max_buffered_entries: int

    def __post_init__(self) -> None:
        if any(not _strict_int(value, minimum=1) for value in (
                self.max_domains, self.max_padded_fft_cells,
                self.max_shift_positions, self.max_overlap_cell_visits,
                self.max_buffered_entries)):
            raise ValueError("shift-domain budget limits must be positive exact integers")

    def to_record(self) -> dict[str, int]:
        return {
            "max_domains": self.max_domains,
            "max_padded_fft_cells": self.max_padded_fft_cells,
            "max_shift_positions": self.max_shift_positions,
            "max_overlap_cell_visits": self.max_overlap_cell_visits,
            "max_buffered_entries": self.max_buffered_entries,
        }


@dataclass(frozen=True, slots=True)
class ShiftDomainWork:
    """Planned or consumed logical work for a complete request."""

    domains: int
    padded_fft_cells: int
    shift_positions: int
    overlap_cell_visits: int
    buffered_entries: int

    def __post_init__(self) -> None:
        if any(not _strict_int(value) for value in (
                self.domains, self.padded_fft_cells, self.shift_positions,
                self.overlap_cell_visits, self.buffered_entries)):
            raise ValueError("shift-domain work counters must be non-negative integers")

    def to_record(self) -> dict[str, int]:
        return {
            "domains": self.domains,
            "padded_fft_cells": self.padded_fft_cells,
            "shift_positions": self.shift_positions,
            "overlap_cell_visits": self.overlap_cell_visits,
            "buffered_entries": self.buffered_entries,
        }


@dataclass(frozen=True, slots=True)
class ShiftDomainEntry:
    row_shift: int
    column_shift: int
    surface_value: float
    overlap_cells: int
    overlap_fraction: float

    def __post_init__(self) -> None:
        if (type(self.row_shift) is not int or type(self.column_shift) is not int
                or type(self.surface_value) not in (int, float)
                or not math.isfinite(float(self.surface_value))
                or not _strict_int(self.overlap_cells, minimum=1)
                or type(self.overlap_fraction) not in (int, float)
                or not math.isfinite(float(self.overlap_fraction))
                or not 0.0 < self.overlap_fraction <= 1.0):
            raise ValueError("shift-domain entry fields are invalid")

    def to_record(self) -> dict[str, int | float]:
        return {
            "row_shift": self.row_shift,
            "column_shift": self.column_shift,
            "surface_value": float(self.surface_value),
            "overlap_cells": self.overlap_cells,
            "overlap_fraction": float(self.overlap_fraction),
        }


def _match_domain_replay_payload(value: "MatchShiftDomain") -> dict[str, Any]:
    return {
        "schema": value.schema,
        "version": value.version,
        "correlation_algorithm_id": value.correlation_algorithm_id,
        "match_identity": _identity_record(value.match_identity),
        "max_shift_cells": value.max_shift_cells,
        "right_grid_shape": list(value.right_grid_shape),
        "min_overlap_cells": value.min_overlap_cells,
        "zero_shift_overlap_cells": value.zero_shift_overlap_cells,
        "shift_overlap_fraction_denominator_cells":
            value.shift_overlap_fraction_denominator_cells,
        "admitted_top_shift_rc": list(value.admitted_top_shift_rc),
        "admitted_peak": float(value.admitted_peak),
        "admitted_second_peak": float(value.admitted_second_peak),
        "admitted_psr": (None if value.admitted_psr_positive_infinity
                          else float(value.admitted_psr)),
        "admitted_psr_positive_infinity": value.admitted_psr_positive_infinity,
        "admitted_overlap_cells": value.admitted_overlap_cells,
        "admitted_score": float(value.admitted_score),
        "top_surface_value": float(value.top_surface_value),
        "top_shift_overlap_cells": value.top_shift_overlap_cells,
        "top_shift_overlap_fraction": float(value.top_shift_overlap_fraction),
        "top_shift_retained": value.top_shift_retained,
        "correlation_surface_sha256": value.correlation_surface_sha256,
        "entries": [entry.to_record() for entry in value.entries],
        "left_signature_geometry_sha256": value.left_signature_geometry_sha256,
        "right_signature_geometry_sha256": value.right_signature_geometry_sha256,
        "cell_homography_sha256": value.cell_homography_sha256,
        "left_signature_content_sha256": value.left_signature_content_sha256,
        "right_signature_content_sha256": value.right_signature_content_sha256,
    }


@dataclass(frozen=True, slots=True)
class MatchShiftDomain:
    schema: str
    version: int
    correlation_algorithm_id: str
    match_identity: plane_axis_support.C1MatchIdentity
    max_shift_cells: int
    right_grid_shape: tuple[int, int]
    min_overlap_cells: int
    zero_shift_overlap_cells: int
    shift_overlap_fraction_denominator_cells: int
    admitted_top_shift_rc: tuple[int, int]
    admitted_peak: float
    admitted_second_peak: float
    admitted_psr: float | None
    admitted_psr_positive_infinity: bool
    admitted_overlap_cells: int
    admitted_score: float
    top_surface_value: float
    top_shift_overlap_cells: int
    top_shift_overlap_fraction: float
    top_shift_retained: bool
    correlation_surface_sha256: str
    entries: tuple[ShiftDomainEntry, ...]
    left_signature_geometry_sha256: str
    right_signature_geometry_sha256: str
    cell_homography_sha256: str
    left_signature_content_sha256: str
    right_signature_content_sha256: str
    replay_sha256: str

    def __post_init__(self) -> None:
        if (self.schema != SHIFT_DOMAIN_SCHEMA
                or self.version != SHIFT_DOMAIN_VERSION
                or type(self.version) is not int
                or self.correlation_algorithm_id
                != plane_graph.PLANE_CORRELATION_ALGORITHM):
            raise ValueError("shift-domain provenance is unsupported")
        if not isinstance(self.match_identity, plane_axis_support.C1MatchIdentity):
            raise ValueError("shift domain requires an exact C1 match identity")
        if not _strict_int(self.max_shift_cells, minimum=1):
            raise ValueError("shift-domain radius must be a positive exact integer")
        if (not isinstance(self.right_grid_shape, tuple)
                or len(self.right_grid_shape) != 2
                or any(not _strict_int(value, minimum=1)
                       for value in self.right_grid_shape)):
            raise ValueError("right-grid shape must be two positive integers")
        if (type(self.min_overlap_cells) is not int
                or self.min_overlap_cells != plane_graph.MIN_OVERLAP_CELLS
                or not _strict_int(self.zero_shift_overlap_cells)
                or not _strict_int(self.shift_overlap_fraction_denominator_cells,
                                   minimum=1)):
            raise ValueError("shift-domain overlap provenance is invalid")
        if (not isinstance(self.admitted_top_shift_rc, tuple)
                or len(self.admitted_top_shift_rc) != 2
                or any(type(value) is not int for value in self.admitted_top_shift_rc)
                or any(abs(value) > self.max_shift_cells
                       for value in self.admitted_top_shift_rc)):
            raise ValueError("admitted top shift is outside the declared radius")
        for number in (self.admitted_peak, self.admitted_second_peak,
                       self.top_surface_value, self.top_shift_overlap_fraction,
                       self.admitted_score):
            if type(number) not in (int, float) or not math.isfinite(float(number)):
                raise ValueError("public shift-domain diagnostics must be finite")
        if (type(self.admitted_psr_positive_infinity) is not bool
                or (self.admitted_psr_positive_infinity and self.admitted_psr is not None)
                or (not self.admitted_psr_positive_infinity
                    and (type(self.admitted_psr) not in (int, float)
                         or not math.isfinite(float(self.admitted_psr))))):
            raise ValueError("admitted PSR must be finite or tagged positive infinity")
        if (self.admitted_peak != self.top_surface_value
                or not _strict_int(self.admitted_overlap_cells,
                                   minimum=self.min_overlap_cells)
                or self.admitted_overlap_cells != self.top_shift_overlap_cells
                or not _strict_int(self.top_shift_overlap_cells)
                or type(self.top_shift_retained) is not bool
                or not 0.0 <= self.top_shift_overlap_fraction <= 1.0
                or self.top_shift_overlap_fraction
                != self.top_shift_overlap_cells
                / self.shift_overlap_fraction_denominator_cells):
            raise ValueError("admitted and recomputed top diagnostics disagree")
        for digest in (
                self.correlation_surface_sha256,
                self.left_signature_geometry_sha256,
                self.right_signature_geometry_sha256,
                self.cell_homography_sha256,
                self.left_signature_content_sha256,
                self.right_signature_content_sha256,
                self.replay_sha256):
            if not _valid_digest(digest):
                raise ValueError("shift-domain provenance must use lowercase SHA-256")
        if (not isinstance(self.entries, tuple)
                or any(not isinstance(entry, ShiftDomainEntry) for entry in self.entries)
                or not self.entries
                or tuple(sorted(self.entries,
                                key=lambda entry: (entry.row_shift, entry.column_shift)))
                != self.entries
                or len({(entry.row_shift, entry.column_shift)
                        for entry in self.entries}) != len(self.entries)):
            raise ValueError("shift-domain entries must be nonempty, unique and row-major")
        entry_by_shift = {(entry.row_shift, entry.column_shift): entry
                          for entry in self.entries}
        if (any(abs(entry.row_shift) > self.max_shift_cells
                or abs(entry.column_shift) > self.max_shift_cells
                or entry.overlap_cells < self.min_overlap_cells
                or entry.overlap_fraction
                != entry.overlap_cells
                / self.shift_overlap_fraction_denominator_cells
                for entry in self.entries)
                ):
            raise ValueError("shift-domain entries violate radius/support invariants")
        top_entry = entry_by_shift.get(self.admitted_top_shift_rc)
        if self.top_shift_retained:
            if (top_entry is None
                    or top_entry.surface_value != self.top_surface_value
                    or top_entry.overlap_cells != self.top_shift_overlap_cells
                    or top_entry.overlap_fraction != self.top_shift_overlap_fraction):
                raise ValueError("retained admitted peak disagrees with its domain entry")
        elif (top_entry is not None
              or self.top_shift_overlap_cells >= self.min_overlap_cells):
            raise ValueError("unretained admitted peak must fail the support floor")
        if _sha256_json(_match_domain_replay_payload(self)) != self.replay_sha256:
            raise ValueError("shift-domain replay digest does not bind its contents")

    def to_record(self) -> dict[str, Any]:
        record = _match_domain_replay_payload(self)
        record["replay_sha256"] = self.replay_sha256
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return record


@dataclass(frozen=True, slots=True)
class ShiftDomainRefusal:
    match_identity: plane_axis_support.C1MatchIdentity | None
    reason: str
    detail: str

    def __post_init__(self) -> None:
        if (self.match_identity is not None
                and not isinstance(self.match_identity,
                                   plane_axis_support.C1MatchIdentity)):
            raise ValueError("shift-domain refusal identity is invalid")
        if (type(self.reason) is not str or self.reason not in SHIFT_DOMAIN_REFUSAL_REASONS
                or type(self.detail) is not str or len(self.detail) > _MAX_DETAIL_LENGTH
                or any(ord(character) < 32 for character in self.detail)):
            raise ValueError("shift-domain refusal fields are invalid")

    def to_record(self) -> dict[str, Any]:
        return {
            "match_identity": (None if self.match_identity is None
                               else _identity_record(self.match_identity)),
            "reason": self.reason,
            "detail": self.detail,
        }


def _input_digest_payload(
    identities: tuple[plane_axis_support.C1MatchIdentity, ...],
    fingerprints: tuple[tuple[plane_axis_support.C1MatchIdentity, str], ...],
) -> dict[str, Any]:
    return {
        "requested_match_identities": [_identity_record(value) for value in identities],
        "request_fingerprints": [
            {"match_identity": _identity_record(identity), "sha256": digest}
            for identity, digest in fingerprints],
    }


@dataclass(frozen=True, slots=True)
class ShiftDomainResult:
    schema: str
    version: int
    correlation_algorithm_id: str
    domains: tuple[MatchShiftDomain, ...]
    refusals: tuple[ShiftDomainRefusal, ...]
    budget: ShiftDomainBudget
    requested_match_identities: tuple[plane_axis_support.C1MatchIdentity, ...]
    request_fingerprints: tuple[tuple[plane_axis_support.C1MatchIdentity, str], ...]
    planned_work: ShiftDomainWork
    consumed_work: ShiftDomainWork
    input_sha256: str
    status: str
    budget_failure_reason: str | None

    def __post_init__(self) -> None:
        if (self.schema != SHIFT_DOMAIN_SCHEMA
                or type(self.version) is not int
                or self.version != SHIFT_DOMAIN_VERSION
                or self.correlation_algorithm_id
                != plane_graph.PLANE_CORRELATION_ALGORITHM):
            raise ValueError("shift-domain result provenance is unsupported")
        if not isinstance(self.budget, ShiftDomainBudget):
            raise ValueError("shift-domain result requires its explicit budget")
        if (not isinstance(self.domains, tuple)
                or any(not isinstance(value, MatchShiftDomain) for value in self.domains)
                or not isinstance(self.refusals, tuple)
                or any(not isinstance(value, ShiftDomainRefusal) for value in self.refusals)
                or not isinstance(self.requested_match_identities, tuple)
                or any(not isinstance(value, plane_axis_support.C1MatchIdentity)
                       for value in self.requested_match_identities)
                or tuple(sorted(set(self.requested_match_identities), key=_identity_key))
                != self.requested_match_identities):
            raise ValueError("shift-domain result collections are not canonical")
        if (not isinstance(self.request_fingerprints, tuple)
                or len(self.request_fingerprints) != len(self.requested_match_identities)
                or tuple(identity for identity, _digest in self.request_fingerprints)
                != self.requested_match_identities
                or any(not _valid_digest(digest)
                       for _identity, digest in self.request_fingerprints)):
            raise ValueError("shift-domain request fingerprints are incomplete")
        if not isinstance(self.planned_work, ShiftDomainWork) or not isinstance(
                self.consumed_work, ShiftDomainWork):
            raise ValueError("shift-domain result work counters are invalid")
        if not _valid_digest(self.input_sha256):
            raise ValueError("shift-domain input digest is invalid")
        expected_input = _sha256_json(_input_digest_payload(
            self.requested_match_identities, self.request_fingerprints))
        if expected_input != self.input_sha256:
            raise ValueError("shift-domain input digest does not bind its request")
        if self.status not in {"complete", "refused", "budget_exhausted"}:
            raise ValueError("shift-domain result status is invalid")
        if self.planned_work.domains > self.budget.max_domains:
            planned_over_budget = True
        else:
            planned_over_budget = any((
                self.planned_work.padded_fft_cells > self.budget.max_padded_fft_cells,
                self.planned_work.shift_positions > self.budget.max_shift_positions,
                self.planned_work.overlap_cell_visits > self.budget.max_overlap_cell_visits,
                self.planned_work.buffered_entries > self.budget.max_buffered_entries,
            ))
        consumed_values = self.consumed_work.to_record()
        planned_values = self.planned_work.to_record()
        budget_limits = {
            "domains": self.budget.max_domains,
            "padded_fft_cells": self.budget.max_padded_fft_cells,
            "shift_positions": self.budget.max_shift_positions,
            "overlap_cell_visits": self.budget.max_overlap_cell_visits,
            "buffered_entries": self.budget.max_buffered_entries,
        }
        if any(consumed_values[key] > budget_limits[key] for key in budget_limits):
            raise ValueError("consumed shift-domain work exceeds its explicit budget")
        if any(consumed_values[key] > planned_values[key] for key in planned_values):
            raise ValueError("consumed shift-domain work exceeds preflight planning")
        domain_ids = tuple(value.match_identity for value in self.domains)
        if (len(set(domain_ids)) != len(domain_ids)
                or tuple(sorted(domain_ids, key=_identity_key)) != domain_ids):
            raise ValueError("shift domains must be unique and identity-sorted")
        refusal_keys = tuple((
            () if value.match_identity is None else _identity_key(value.match_identity),
            value.reason, value.detail) for value in self.refusals)
        if tuple(sorted(refusal_keys)) != refusal_keys:
            raise ValueError("shift-domain refusals must be deterministically sorted")
        if self.status == "complete":
            if (self.budget_failure_reason is not None or self.refusals
                    or domain_ids != self.requested_match_identities
                    or not self.requested_match_identities
                    or planned_over_budget):
                raise ValueError("complete result must contain every requested domain only")
            expected_work = _work_for_domains(self.domains)
            if self.planned_work != expected_work:
                raise ValueError("complete result planned work disagrees with its domains")
            expected_consumed = ShiftDomainWork(
                expected_work.domains, expected_work.padded_fft_cells,
                expected_work.shift_positions, expected_work.overlap_cell_visits,
                sum(len(domain.entries) for domain in self.domains))
            if self.consumed_work != expected_consumed:
                raise ValueError("complete result consumed work disagrees with enumeration")
        elif self.status == "refused":
            if (self.domains or self.budget_failure_reason is not None
                    or planned_over_budget):
                raise ValueError("refused result cannot carry domains or budget failure")
            requested_refusals = [value.match_identity for value in self.refusals
                                  if value.match_identity in self.requested_match_identities]
            if self.requested_match_identities and set(requested_refusals) != set(
                    self.requested_match_identities):
                raise ValueError("refused result must examine every requested identity")
            if (len(requested_refusals) != len(set(requested_refusals))
                    or not self.refusals):
                raise ValueError("refused result requires one canonical refusal per match")
        else:
            if (self.domains or self.refusals
                    or self.budget_failure_reason not in {
                        "planned_budget_exceeded", "runtime_budget_discrepancy",
                        "shift_domain_incomplete"}):
                raise ValueError("budget-exhausted result cannot carry partial evidence")
            if (self.budget_failure_reason == "planned_budget_exceeded") != planned_over_budget:
                raise ValueError("planned budget failure status disagrees with its counters")

    def to_record(self) -> dict[str, Any]:
        record = {
            "schema": self.schema,
            "version": self.version,
            "correlation_algorithm_id": self.correlation_algorithm_id,
            "domains": [value.to_record() for value in self.domains],
            "refusals": [value.to_record() for value in self.refusals],
            "budget": self.budget.to_record(),
            "requested_match_identities": [
                _identity_record(value) for value in self.requested_match_identities],
            "request_fingerprints": [
                {"match_identity": _identity_record(identity), "sha256": digest}
                for identity, digest in self.request_fingerprints],
            "planned_work": self.planned_work.to_record(),
            "consumed_work": self.consumed_work.to_record(),
            "input_sha256": self.input_sha256,
            "status": self.status,
            "budget_failure_reason": self.budget_failure_reason,
        }
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return record


@dataclass(slots=True)
class _WorkItem:
    identity: plane_axis_support.C1MatchIdentity
    match: plane_graph.PlaneMatch
    left_original: lattice_planes.PlaneSignature
    right_original: lattice_planes.PlaneSignature
    left: lattice_planes.PlaneSignature
    right: lattice_planes.PlaneSignature
    left_geometry_sha256: str
    right_geometry_sha256: str
    homography_sha256: str
    left_content_sha256: str
    right_content_sha256: str
    homography: np.ndarray
    warped_cells: np.ndarray
    warped_coverage: np.ndarray
    right_coverage: np.ndarray
    fixed_overlap: np.ndarray
    first: np.ndarray
    second: np.ndarray
    fraction_denominator: int


def _region_footprint_parts(value: Any) -> tuple[Any, tuple[bytes, ...]]:
    if value is None:
        return None, ()
    try:
        record = {
            "source_region_id": value.source_region_id,
            "blocks_per_period": float(value.blocks_per_period),
            "origin_uv": list(value.origin_uv),
            "grid_shape": list(value.grid_shape),
            "subsample_factor": value.subsample_factor,
            "total_samples_per_cell": value.total_samples_per_cell,
        }
        return record, (value.inside_region_sample_counts,
                        value.valid_sample_counts)
    except (AttributeError, TypeError, ValueError, OverflowError):
        raise ValueError("signature region footprint is malformed")


def _signature_content_sha256(
    signature: lattice_planes.PlaneSignature,
    plane_index: int,
    canonical_b: float,
) -> str:
    if not isinstance(signature, lattice_planes.PlaneSignature):
        raise ValueError("signature type is invalid")
    arrays = []
    for name in ("cells", "interior", "coverage"):
        source = getattr(signature, name)
        if not isinstance(source, np.ndarray) or source.dtype.hasobject:
            raise ValueError("signature arrays must be non-object NumPy arrays")
        array = np.ascontiguousarray(source)
        arrays.append((name, array))
    cells, interior, coverage = (item[1] for item in arrays)
    if (cells.ndim != 2 or cells.size == 0 or interior.shape != cells.shape
            or coverage.shape != cells.shape
            or not np.issubdtype(cells.dtype, np.number)
            or not np.issubdtype(interior.dtype, np.number)
            or coverage.dtype != np.dtype(bool)):
        raise ValueError("signature array shapes or dtypes are invalid")
    if (type(plane_index) is not int or plane_index < 0
            or type(canonical_b) not in (int, float)
            or not math.isfinite(float(canonical_b)) or canonical_b <= 0.0):
        raise ValueError("signature canonical identity is invalid")
    reading = signature.reading
    if not isinstance(reading, lattice_planes.PlaneReading):
        raise ValueError("signature reading is invalid")
    origin = tuple(float(value) for value in signature.origin_uv)
    if len(origin) != 2 or not all(math.isfinite(value) for value in origin):
        raise ValueError("signature origin is invalid")
    signature_b = float(signature.blocks_per_period)
    if not math.isfinite(signature_b) or signature_b <= 0.0:
        raise ValueError("signature b value is invalid")
    if reading.matching_alternatives(canonical_b) != reading.matching_alternatives(
            signature_b):
        raise ValueError("signature b does not bind its canonical reading alternative")
    try:
        order_key = lattice_planes.reading_order_key(reading)
        order_record = _json_value(order_key)
    except (AttributeError, TypeError, ValueError, OverflowError):
        raise ValueError("ordered reading identity is malformed")
    array_records = []
    for name, array in arrays:
        array_records.append({
            "name": name,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        })
    footprint_record, footprint_bytes = _region_footprint_parts(
        signature.region_footprint)
    payload = {
        "arrays": array_records,
        "origin_uv": list(origin),
        "canonical_blocks_per_period": float(canonical_b),
        "signature_blocks_per_period": signature_b,
        "ordered_reading_identity": {
            "frame": reading.frame,
            "plane_index": plane_index,
            "reading_order_key": order_record,
        },
        "region_footprint": footprint_record,
        "region_footprint_byte_lengths": [len(value) for value in footprint_bytes],
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json(payload).encode("ascii"))
    for (name, array) in arrays:
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    for packed in footprint_bytes:
        if type(packed) is not bytes:
            raise ValueError("signature region footprint bytes are invalid")
        digest.update(len(packed).to_bytes(8, "big", signed=False))
        digest.update(packed)
    return digest.hexdigest()


def _snapshot_signature(
    signature: lattice_planes.PlaneSignature,
) -> lattice_planes.PlaneSignature:
    if not isinstance(signature, lattice_planes.PlaneSignature):
        raise ValueError("signature type is invalid")
    arrays = []
    for name in ("cells", "interior", "coverage"):
        source = getattr(signature, name)
        if not isinstance(source, np.ndarray) or source.dtype.hasobject:
            raise ValueError("signature arrays must be non-object NumPy arrays")
        arrays.append(np.array(source, copy=True, order="C"))
    return lattice_planes.PlaneSignature(
        reading=signature.reading,
        blocks_per_period=signature.blocks_per_period,
        cells=arrays[0], interior=arrays[1], coverage=arrays[2],
        origin_uv=tuple(signature.origin_uv),
        region_footprint=signature.region_footprint)


def _candidate_signatures(
    identity: plane_axis_support.C1MatchIdentity,
    match: plane_graph.PlaneMatch,
    readings: Sequence[lattice_planes.PlaneReading],
    signatures: Sequence[lattice_planes.PlaneSignature],
) -> tuple[lattice_planes.PlaneSignature, lattice_planes.PlaneSignature]:
    try:
        read_by_index, index_by_object, _frames = plane_graph._canonical_reading_index(
            readings)
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise _EvidenceError(
            "signature_identity_mismatch", "complete reading universe is invalid") from error
    signature_rows: list[lattice_planes.PlaneSignature] = []
    for signature in signatures:
        if not isinstance(signature, lattice_planes.PlaneSignature):
            raise _EvidenceError("signature_identity_mismatch", "signature object type is invalid")
        reading = signature.reading
        if not isinstance(reading, lattice_planes.PlaneReading):
            raise _EvidenceError("signature_identity_mismatch", "signature reading type is invalid")
        if not plane_graph._is_finite_positive_number(signature.blocks_per_period):
            raise _EvidenceError(
                "signature_identity_mismatch", "signature b is not finite and positive")
        reading_index = index_by_object.get(id(reading))
        if reading_index is None or read_by_index.get(reading_index) is not reading:
            raise _EvidenceError(
                "signature_identity_mismatch", "signature reading is outside complete universe")
        signature_rows.append(signature)

    def find(frame: str, plane_index: int, axis: int, b_value: float):
        reading = read_by_index.get((frame, plane_index))
        if reading is None:
            raise _EvidenceError("signature_missing", "addressed reading has no signature")
        if reading.frame != frame or reading.axis != axis:
            raise _EvidenceError("signature_identity_mismatch", "reading identity differs from C1")
        alternatives = reading.matching_alternatives(b_value)
        if len(alternatives) != 1:
            raise _EvidenceError("signature_identity_mismatch", "match b is not uniquely retained")
        canonical_b = float(alternatives[0].blocks_per_period)
        found = []
        for signature in signature_rows:
            if signature.reading is not reading:
                continue
            selected = reading.matching_alternatives(canonical_b)
            candidate = reading.matching_alternatives(signature.blocks_per_period)
            if (len(selected) == 1 and len(candidate) == 1
                    and candidate[0].blocks_per_period == selected[0].blocks_per_period):
                found.append(signature)
        if not found:
            raise _EvidenceError("signature_missing", "selected reading signature is absent")
        if len(found) != 1:
            raise _EvidenceError("signature_ambiguous", "selected reading signature is duplicated")
        return found[0], canonical_b

    left, left_b = find(identity.left_frame, identity.left_plane,
                        identity.left_axis, match.left_b)
    right, right_b = find(identity.right_frame, identity.right_plane,
                          identity.right_axis, match.right_b)
    if (left.reading.axis != identity.left_axis
            or right.reading.axis != identity.right_axis):
        raise _EvidenceError("signature_identity_mismatch", "signature axes differ from C1")
    return left, right


class _EvidenceError(Exception):
    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail[:_MAX_DETAIL_LENGTH]


def _clipped_area(rows: int, columns: int, row_shift: int,
                  column_shift: int) -> int:
    return (max(0, rows - abs(row_shift))
            * max(0, columns - abs(column_shift)))


def _axis_visit_sum(length: int, radius: int) -> int:
    # Sum max(0, length - abs(shift)) for every shift in [-radius, radius].
    extent = min(radius, length - 1)
    return length + 2 * (extent * length - extent * (extent + 1) // 2)


def _planned_work(items: Sequence[_WorkItem], radius: int,
                  requested_domains: int | None = None) -> ShiftDomainWork:
    domains = len(items) if requested_domains is None else requested_domains
    padded = positions = visits = 0
    for item in items:
        rows, columns = item.right.cells.shape
        padded += (rows + 6 * radius) * (columns + 6 * radius)
        span = 2 * radius + 1
        positions += span * span
        visits += span * span * (rows + 2 * radius) * (columns + 2 * radius)
    return ShiftDomainWork(domains, padded, positions, visits, positions)


def _work_for_domains(domains: Sequence[MatchShiftDomain]) -> ShiftDomainWork:
    padded = positions = visits = buffered = 0
    for domain in domains:
        rows, columns = domain.right_grid_shape
        radius = domain.max_shift_cells
        padded += (rows + 6 * radius) * (columns + 6 * radius)
        span = 2 * radius + 1
        positions += span * span
        visits += span * span * (rows + 2 * radius) * (columns + 2 * radius)
        # The budgeted maximum is the full enumerated square, not the
        # support-pruned number of entries ultimately retained.
        buffered += span * span
    return ShiftDomainWork(len(domains), padded, positions, visits, buffered)


def _budget_exceeds(work: ShiftDomainWork, budget: ShiftDomainBudget) -> bool:
    return any((
        work.domains > budget.max_domains,
        work.padded_fft_cells > budget.max_padded_fft_cells,
        work.shift_positions > budget.max_shift_positions,
        work.overlap_cell_visits > budget.max_overlap_cell_visits,
        work.buffered_entries > budget.max_buffered_entries,
    ))


def _match_record(match: plane_graph.PlaneMatch) -> dict[str, Any]:
    return {
        "left": match.left,
        "right": match.right,
        "left_axis": match.left_axis,
        "right_axis": match.right_axis,
        "left_b": _float_record(match.left_b),
        "right_b": _float_record(match.right_b),
        "quadrant_of_right": match.quadrant_of_right,
        "psr": _float_record(match.psr),
        "peak": _float_record(match.peak),
        "second_peak": _float_record(match.second_peak),
        "overlap_cells": match.overlap_cells,
        "integer_residual": (None if match.integer_residual is None
                              else _float_record(match.integer_residual)),
        "score": _float_record(match.score),
        "left_plane": match.left_plane,
        "right_plane": match.right_plane,
        "cell_shift_rc": (None if match.cell_shift_rc is None
                          else list(match.cell_shift_rc)),
    }


def _c1_axis_records(
    values: Sequence[plane_axis_support.MappedCellShiftAxis],
) -> list[dict[str, Any]]:
    return [value.to_record() for value in sorted(
        values, key=lambda item: (item.match_identity.sort_key(), item.right_local_axis))]


def _request_fingerprint(
    identity: plane_axis_support.C1MatchIdentity,
    radius: Any,
    match_candidates: Sequence[plane_graph.PlaneMatch],
    axes: Sequence[plane_axis_support.MappedCellShiftAxis],
    content_hashes: Sequence[str],
    refusal: tuple[str, str] | None,
) -> str:
    payload: dict[str, Any] = {
        "match_identity": _identity_record(identity),
        "radius": (radius if type(radius) is int else {"invalid_type": type(radius).__name__}),
        "matches": sorted((_match_record(match) for match in match_candidates),
                          key=_canonical_json),
        "c1_mapped_axes": _c1_axis_records(axes),
        "signature_content_sha256": sorted(content_hashes),
        "refusal": None if refusal is None else {
            "reason": refusal[0], "detail": refusal[1]},
    }
    return _sha256_json(payload)


def _domain_digest(domain_values: dict[str, Any]) -> str:
    return _sha256_json(domain_values)


def _shift_support(mask: np.ndarray, row_shift: int,
                   column_shift: int) -> int:
    rows, columns = mask.shape
    row_start = max(0, -row_shift)
    row_stop = min(rows, rows - row_shift)
    column_start = max(0, -column_shift)
    column_stop = min(columns, columns - column_shift)
    if row_stop <= row_start or column_stop <= column_start:
        return 0
    first = mask[row_start:row_stop, column_start:column_stop]
    second = mask[row_start + row_shift:row_stop + row_shift,
                   column_start + column_shift:column_stop + column_shift]
    return int(np.count_nonzero(first & second))


def _correlation_replay_ok(match: plane_graph.PlaneMatch,
                           surface: np.ndarray,
                           support_cells: np.ndarray,
                           item: _WorkItem,
                           radius: int) -> tuple[bool, tuple[float, float, float, tuple[int, int]]]:
    peak, psr, second, shift = plane_graph._peak_statistics(
        surface, max_shift_cells=radius,
        admissible=support_cells >= plane_graph.MIN_OVERLAP_CELLS)
    if not isinstance(match.cell_shift_rc, tuple) or match.cell_shift_rc != shift:
        return False, (peak, psr, second, shift)
    psr_equal = ((math.isinf(psr) and psr > 0.0
                  and type(match.psr) in (int, float)
                  and math.isinf(float(match.psr)) and float(match.psr) > 0.0)
                 or (math.isfinite(psr) and type(match.psr) in (int, float)
                     and math.isfinite(float(match.psr)) and float(match.psr) == psr))
    share = min(float(item.left.reading.structural_share),
                float(item.right.reading.structural_share))
    valid_cells = min(int(np.count_nonzero(item.warped_coverage)),
                      int(np.count_nonzero(item.right_coverage)))
    selected_overlap = int(support_cells[shift[0] + radius, shift[1] + radius])
    overlap_fraction = (selected_overlap / valid_cells
                        if valid_cells else 0.0)
    margin = (peak - second) / peak if peak > 1e-12 else 0.0
    score = (0.5 * min(psr / plane_graph.PSR_SCALE, 1.0)
             + 0.3 * overlap_fraction
             + 0.2 * float(np.clip(margin, 0.0, 1.0))
             - 0.5 * (1.0 - share))
    stats_equal = (
        type(match.peak) in (int, float) and float(match.peak) == peak
        and type(match.second_peak) in (int, float)
        and float(match.second_peak) == second
        and psr_equal
        and type(match.overlap_cells) is int
        and match.overlap_cells == selected_overlap
        and type(match.score) in (int, float)
        and math.isfinite(float(match.score)) and float(match.score) == score
    )
    return stats_equal, (peak, psr, second, shift)


def _prepare_item(
    identity: plane_axis_support.C1MatchIdentity,
    match: plane_graph.PlaneMatch,
    axes: Sequence[plane_axis_support.MappedCellShiftAxis],
    readings: Sequence[lattice_planes.PlaneReading],
    signatures: Sequence[lattice_planes.PlaneSignature],
    radius: int,
) -> tuple[_WorkItem | None, tuple[str, str] | None, tuple[str, ...]]:
    content_hashes: list[str] = []
    try:
        left_original, right_original = _candidate_signatures(
            identity, match, readings, signatures)
    except _EvidenceError as error:
        return None, (error.reason, error.detail), ()  # type: ignore[return-value]
    try:
        left_snapshot = _snapshot_signature(left_original)
        right_snapshot = _snapshot_signature(right_original)
        left_b = float(left_snapshot.reading.matching_alternatives(
            match.left_b)[0].blocks_per_period)
        right_b = float(right_snapshot.reading.matching_alternatives(
            match.right_b)[0].blocks_per_period)
        left_content = _signature_content_sha256(
            left_snapshot, identity.left_plane, left_b)
        right_content = _signature_content_sha256(
            right_snapshot, identity.right_plane, right_b)
        content_hashes.extend((left_content, right_content))
    except (AttributeError, IndexError, TypeError, ValueError, OverflowError) as error:
        return None, ("signature_content_invalid", str(error)[:_MAX_DETAIL_LENGTH]), tuple(content_hashes)
    try:
        left_geometry = plane_axis_support._signature_geometry_digest(
            left_snapshot, identity.left_plane)
        right_geometry = plane_axis_support._signature_geometry_digest(
            right_snapshot, identity.right_plane)
        if left_geometry is None or right_geometry is None:
            return None, ("homography_invalid", "signature geometry is not finite"), tuple(content_hashes)
        homography = plane_graph.cell_grid_homography(
            left_snapshot, right_snapshot, match.quadrant_of_right)
        homography_digest = plane_axis_support._grid_homography_digest(
            left_snapshot, right_snapshot, match.quadrant_of_right,
            match.left_plane, match.right_plane, homography)
        if homography_digest is None:
            return None, ("homography_invalid", "normalized homography is invalid"), tuple(content_hashes)
    except (AttributeError, TypeError, ValueError, OverflowError, np.linalg.LinAlgError) as error:
        return None, ("homography_invalid", str(error)[:_MAX_DETAIL_LENGTH]), tuple(content_hashes)
    if len(axes) != 2 or {value.right_local_axis for value in axes} != {"u", "v"}:
        return None, ("provenance_mismatch", "C1 does not provide a complete u/v pair"), tuple(content_hashes)
    if any((value.left_signature_geometry_sha256 != left_geometry
            or value.right_signature_geometry_sha256 != right_geometry
            or value.cell_homography_sha256 != homography_digest)
           for value in axes):
        return None, ("provenance_mismatch", "current signature geometry or homography differs from C1"), tuple(content_hashes)
    try:
        warped_cells, warped_coverage, right_cells, right_coverage = plane_graph._common_canvas_inputs(
            left_snapshot, right_snapshot, homography, radius,
            warp=lattice_planes.warp_signature)
        if warped_coverage.shape != right_coverage.shape:
            return None, ("signature_content_invalid", "warped coverage shape differs"), tuple(content_hashes)
        fixed = warped_coverage & right_coverage
        first = np.where(warped_coverage, warped_cells, 0.0).astype(np.float64)
        second = np.where(right_coverage, right_cells, 0.0).astype(np.float64)
        first_total = float(first.sum())
        second_total = float(second.sum())
        if not math.isfinite(first_total) or not math.isfinite(second_total):
            return None, ("correlation_nonfinite", "fixed correlation map sum is non-finite"), tuple(content_hashes)
        if first_total <= 0.0 or second_total <= 0.0:
            return None, ("correlation_replay_mismatch", "admitted signature pair has no correlation signal"), tuple(content_hashes)
        denominator = min(int(np.count_nonzero(warped_coverage)),
                          int(np.count_nonzero(right_coverage)))
        if denominator <= 0:
            return None, ("signature_content_invalid", "overlap fraction denominator is empty"), tuple(content_hashes)
    except (AttributeError, IndexError, TypeError, ValueError, OverflowError) as error:
        return None, ("homography_invalid", str(error)[:_MAX_DETAIL_LENGTH]), tuple(content_hashes)
    item = _WorkItem(
        identity, match, left_original, right_original,
        left_snapshot, right_snapshot, left_geometry, right_geometry,
        homography_digest, left_content, right_content, np.asarray(homography),
        warped_cells, warped_coverage, right_coverage, fixed, first, second,
        denominator)
    return item, None, tuple(content_hashes)


def _make_domain(item: _WorkItem, radius: int, surface: np.ndarray,
                 support_cells: np.ndarray,
                 replay: tuple[float, float, float, tuple[int, int]]) -> MatchShiftDomain:
    rows, columns = surface.shape
    if surface.shape != (2 * radius + 1, 2 * radius + 1) or not np.isfinite(surface).all():
        raise _EvidenceError("correlation_nonfinite", "correlation surface is incomplete or non-finite")
    peak, psr, second_peak, top_shift = replay
    entries = []
    top_count = None
    span = 2 * radius + 1
    for row_shift in range(-radius, radius + 1):
        for column_shift in range(-radius, radius + 1):
            count = int(support_cells[row_shift + radius, column_shift + radius])
            if (row_shift, column_shift) == top_shift:
                top_count = count
            if count < plane_graph.MIN_OVERLAP_CELLS:
                continue
            entries.append(ShiftDomainEntry(
                row_shift, column_shift,
                float(surface[row_shift + radius, column_shift + radius]),
                count, count / item.fraction_denominator))
    if top_count is None:
        raise _EvidenceError("shift_domain_incomplete", "admitted top shift was not enumerated")
    top_fraction = top_count / item.fraction_denominator
    match = item.match
    psr_infinite = math.isinf(psr) and psr > 0.0
    replay_fields = {
        "schema": SHIFT_DOMAIN_SCHEMA,
        "version": SHIFT_DOMAIN_VERSION,
        "correlation_algorithm_id": plane_graph.PLANE_CORRELATION_ALGORITHM,
        "match_identity": item.identity,
        "max_shift_cells": radius,
        "right_grid_shape": tuple(item.right.cells.shape),
        "min_overlap_cells": plane_graph.MIN_OVERLAP_CELLS,
        "zero_shift_overlap_cells": int(np.count_nonzero(item.fixed_overlap)),
        "shift_overlap_fraction_denominator_cells": item.fraction_denominator,
        "admitted_top_shift_rc": top_shift,
        "admitted_peak": peak,
        "admitted_second_peak": second_peak,
        "admitted_psr": None if psr_infinite else psr,
        "admitted_psr_positive_infinity": psr_infinite,
        "admitted_overlap_cells": int(match.overlap_cells),
        "admitted_score": float(match.score),
        "top_surface_value": float(surface[top_shift[0] + radius,
                                            top_shift[1] + radius]),
        "top_shift_overlap_cells": top_count,
        "top_shift_overlap_fraction": top_fraction,
        "top_shift_retained": top_count >= plane_graph.MIN_OVERLAP_CELLS,
        "correlation_surface_sha256": _sha256_bytes(
            np.asarray(surface, dtype="<f8").tobytes(order="C")),
        "entries": tuple(entries),
        "left_signature_geometry_sha256": item.left_geometry_sha256,
        "right_signature_geometry_sha256": item.right_geometry_sha256,
        "cell_homography_sha256": item.homography_sha256,
        "left_signature_content_sha256": item.left_content_sha256,
        "right_signature_content_sha256": item.right_content_sha256,
    }
    temporary = object.__new__(MatchShiftDomain)
    for name, value in replay_fields.items():
        object.__setattr__(temporary, name, value)
    payload = _match_domain_replay_payload(temporary)
    digest = _domain_digest(payload)
    return MatchShiftDomain(**replay_fields, replay_sha256=digest)


def _result(
    *, domains: Sequence[MatchShiftDomain], refusals: Sequence[ShiftDomainRefusal],
    budget: ShiftDomainBudget,
    identities: tuple[plane_axis_support.C1MatchIdentity, ...],
    fingerprints: tuple[tuple[plane_axis_support.C1MatchIdentity, str], ...],
    planned: ShiftDomainWork, consumed: ShiftDomainWork, status: str,
    budget_failure_reason: str | None,
) -> ShiftDomainResult:
    ordered_domains = tuple(sorted(domains, key=lambda value: value.match_identity.sort_key()))
    ordered_refusals = tuple(sorted(refusals, key=lambda value: (
        () if value.match_identity is None else value.match_identity.sort_key(),
        value.reason, value.detail)))
    input_digest = _sha256_json(_input_digest_payload(identities, fingerprints))
    return ShiftDomainResult(
        SHIFT_DOMAIN_SCHEMA, SHIFT_DOMAIN_VERSION,
        plane_graph.PLANE_CORRELATION_ALGORITHM,
        ordered_domains, ordered_refusals, budget, identities, fingerprints,
        planned, consumed, input_digest, status, budget_failure_reason)


def _budget_work(budget: ShiftDomainBudget) -> ShiftDomainWork:
    return ShiftDomainWork(0, 0, 0, 0, 0)


def build_shift_domains(
    support: plane_axis_support.C1AxisSupport,
    matches: Sequence[plane_graph.PlaneMatch],
    signatures: Sequence[lattice_planes.PlaneSignature],
    *,
    readings: Sequence[lattice_planes.PlaneReading],
    max_shift_cells: int,
    budget: ShiftDomainBudget,
) -> ShiftDomainResult:
    """Build all bounded domains for the exact C1 mapped-match set.

    ``readings`` is the same complete canonical plane universe used to produce
    the graph matches. Signature-bearing readings alone cannot safely resolve
    a match index when a signatureless reading sorts before an endpoint.
    No surface is allocated until the entire selected set has been resolved and
    its explicit request-wide budget has passed preflight.
    """
    if not isinstance(budget, ShiftDomainBudget):
        raise ValueError("an explicit ShiftDomainBudget is required")
    if not isinstance(support, plane_axis_support.C1AxisSupport) or support.refusal is not None:
        refusal = ShiftDomainRefusal(None, "provenance_mismatch",
                                     "accepted C1 support is required")
        return _result(
            domains=(), refusals=(refusal,), budget=budget, identities=(),
            fingerprints=(), planned=_budget_work(budget), consumed=_budget_work(budget),
            status="refused", budget_failure_reason=None)
    mapped_by_id: dict[plane_axis_support.C1MatchIdentity,
                       list[plane_axis_support.MappedCellShiftAxis]] = {}
    for axis in support.mapped_cell_shifts:
        mapped_by_id.setdefault(axis.match_identity, []).append(axis)
    identities = tuple(sorted(mapped_by_id, key=_identity_key))
    if not identities:
        refusal = ShiftDomainRefusal(None, "selected_identity_missing",
                                     "C1 contains no complete mapped u/v match")
        return _result(
            domains=(), refusals=(refusal,), budget=budget, identities=(),
            fingerprints=(), planned=_budget_work(budget), consumed=_budget_work(budget),
            status="refused", budget_failure_reason=None)

    if type(max_shift_cells) is not int or max_shift_cells <= 0:
        reason = "shift_domain_incomplete"
        detail = "declared radius must be a positive exact integer"
        fingerprints = tuple((identity, _request_fingerprint(
            identity, max_shift_cells, (), mapped_by_id[identity], (), (reason, detail)))
            for identity in identities)
        refusals = tuple(ShiftDomainRefusal(identity, reason, detail)
                         for identity in identities)
        return _result(
            domains=(), refusals=refusals, budget=budget, identities=identities,
            fingerprints=fingerprints, planned=_budget_work(budget),
            consumed=_budget_work(budget), status="refused", budget_failure_reason=None)

    # Even a one-cell right grid needs this much work. Refuse an impossible
    # radius before signature snapshots or common-canvas allocations.
    span = 2 * max_shift_cells + 1
    minimum = ShiftDomainWork(
        len(identities), len(identities) * (1 + 6 * max_shift_cells) ** 2,
        len(identities) * span ** 2,
        len(identities) * span ** 2 * (1 + 2 * max_shift_cells) ** 2,
        len(identities) * span ** 2)
    if _budget_exceeds(minimum, budget):
        fingerprints = tuple((identity, _request_fingerprint(
            identity, max_shift_cells, (), mapped_by_id[identity], (), None))
            for identity in identities)
        return _result(
            domains=(), refusals=(), budget=budget, identities=identities,
            fingerprints=fingerprints, planned=minimum, consumed=_budget_work(budget),
            status="budget_exhausted", budget_failure_reason="planned_budget_exceeded")

    matches_by_identity: dict[plane_axis_support.C1MatchIdentity, list[plane_graph.PlaneMatch]] = {}
    global_bad_matches = False
    for match in matches if isinstance(matches, Sequence) else ():
        if not isinstance(match, plane_graph.PlaneMatch):
            global_bad_matches = True
            continue
        try:
            identity = plane_axis_support.C1MatchIdentity.from_match(match)
        except (TypeError, ValueError, OverflowError):
            global_bad_matches = True
            continue
        if identity in mapped_by_id:
            matches_by_identity.setdefault(identity, []).append(match)
        # Valid graph hypotheses outside C1's selected mapped set are ignored.
    if not isinstance(matches, Sequence):
        global_bad_matches = True

    sequence_signatures = tuple(signatures) if isinstance(signatures, Sequence) else ()
    global_bad_signatures = not isinstance(signatures, Sequence)
    sequence_readings = tuple(readings) if isinstance(readings, Sequence) else ()
    global_bad_readings = not isinstance(readings, Sequence)
    refusals_by_identity: dict[plane_axis_support.C1MatchIdentity,
                               tuple[str, str]] = {}
    match_candidates: dict[plane_axis_support.C1MatchIdentity,
                           list[plane_graph.PlaneMatch]] = {}
    work_items: list[_WorkItem] = []
    content_by_identity: dict[plane_axis_support.C1MatchIdentity, tuple[str, ...]] = {}

    for identity in identities:
        candidates = matches_by_identity.get(identity, [])
        match_candidates[identity] = list(candidates)
        if len(candidates) == 0:
            refusals_by_identity[identity] = (
                "selected_identity_missing", "C1 selected match was not supplied")
            continue
        if len(candidates) != 1:
            refusals_by_identity[identity] = (
                "selected_identity_ambiguous", "C1 selected match is duplicated")
            continue
        if global_bad_matches:
            refusals_by_identity[identity] = (
                "selected_identity_missing", "supplied match collection contains an invalid row")
            continue
        if global_bad_signatures:
            refusals_by_identity[identity] = (
                "signature_missing", "signature collection is not a sequence")
            continue
        if global_bad_readings:
            refusals_by_identity[identity] = (
                "signature_identity_mismatch", "complete reading universe is not a sequence")
            continue
        item, error, content_hashes = _prepare_item(
            identity, candidates[0], mapped_by_id[identity],
            sequence_readings, sequence_signatures, max_shift_cells)
        content_by_identity[identity] = content_hashes
        if error is not None:
            refusals_by_identity[identity] = error
            continue
        assert item is not None
        work_items.append(item)

    planned = _planned_work(work_items, max_shift_cells, len(identities))
    all_fingerprints = []
    for identity in identities:
        all_fingerprints.append((identity, _request_fingerprint(
            identity, max_shift_cells, match_candidates.get(identity, ()),
            mapped_by_id[identity], content_by_identity.get(identity, ()),
            refusals_by_identity.get(identity))))
    fingerprints = tuple(all_fingerprints)
    if _budget_exceeds(planned, budget):
        return _result(
            domains=(), refusals=(), budget=budget, identities=identities,
            fingerprints=fingerprints, planned=planned, consumed=_budget_work(budget),
            status="budget_exhausted", budget_failure_reason="planned_budget_exceeded")

    # Malformed rows cannot be classified as losing q/b hypotheses. Preserve
    # a request-level refusal while still examining every selected identity.
    foreign_refusals: list[ShiftDomainRefusal] = []
    if global_bad_matches:
        foreign_refusals.append(ShiftDomainRefusal(
            None, "selected_identity_missing", "supplied match collection contains an invalid row"))
    consumed_domains = consumed_padded = consumed_positions = consumed_visits = 0
    consumed_entries = 0
    completed: list[MatchShiftDomain] = []
    runtime_incomplete = False
    runtime_reason = "runtime_budget_discrepancy"
    runtime_refusals: dict[plane_axis_support.C1MatchIdentity, tuple[str, str]] = {}

    for item in work_items:
        rows, columns = item.right.cells.shape
        expected_padded = (rows + 6 * max_shift_cells) * (columns + 6 * max_shift_cells)
        expected_span = 2 * max_shift_cells + 1
        expected_positions = expected_span * expected_span
        if (consumed_padded + expected_padded > budget.max_padded_fft_cells
                or consumed_domains + 1 > budget.max_domains
                or consumed_positions + expected_positions > budget.max_shift_positions):
            runtime_incomplete = True
            break
        correlation = plane_graph._correlation_surface_with_support(
            item.first, item.second, max_shift_cells=max_shift_cells,
            first_support=item.warped_coverage,
            second_support=item.right_coverage)
        consumed_domains += 1
        consumed_padded += expected_padded
        consumed_positions += expected_positions
        if correlation is None or correlation.values.shape != (expected_span, expected_span):
            runtime_refusals[item.identity] = (
                "correlation_nonfinite", "correlation surface is absent or incomplete")
            continue
        surface = correlation.values
        if not np.isfinite(surface).all():
            runtime_refusals[item.identity] = (
                "correlation_nonfinite", "correlation surface contains non-finite values")
            continue
        expected_visits = expected_positions * (rows + 2 * max_shift_cells) * (columns + 2 * max_shift_cells)
        if consumed_visits + expected_visits > budget.max_overlap_cell_visits:
            runtime_incomplete = True
            break
        try:
            replay_ok, replay = _correlation_replay_ok(
                item.match, surface, correlation.support_cells,
                item, max_shift_cells)
        except (AttributeError, TypeError, ValueError, OverflowError, FloatingPointError):
            runtime_refusals[item.identity] = (
                "correlation_replay_mismatch", "stored match diagnostics could not be replayed")
            continue
        if not replay_ok:
            runtime_refusals[item.identity] = (
                "correlation_replay_mismatch", "stored match diagnostics differ from exact replay")
            continue
        try:
            domain = _make_domain(
                item, max_shift_cells, surface, correlation.support_cells, replay)
        except _EvidenceError as error:
            if error.reason == "shift_domain_incomplete":
                runtime_incomplete = True
                runtime_reason = "shift_domain_incomplete"
                break
            runtime_refusals[item.identity] = (error.reason, error.detail)
            continue
        consumed_visits += expected_visits
        consumed_entries += len(domain.entries)
        if consumed_entries > budget.max_buffered_entries:
            runtime_incomplete = True
            break
        completed.append(domain)

    consumed = ShiftDomainWork(
        consumed_domains, consumed_padded, consumed_positions,
        consumed_visits, consumed_entries)
    if runtime_incomplete:
        return _result(
            domains=(), refusals=(), budget=budget, identities=identities,
            fingerprints=fingerprints, planned=planned, consumed=consumed,
            status="budget_exhausted", budget_failure_reason=runtime_reason)

    # Re-hash source arrays after all work. A change during the request cannot
    # be silently paired with a replay made from its snapshot.
    for item in work_items:
        try:
            current_left = _signature_content_sha256(
                item.left_original, item.identity.left_plane,
                float(item.left.reading.matching_alternatives(
                    item.match.left_b)[0].blocks_per_period))
            current_right = _signature_content_sha256(
                item.right_original, item.identity.right_plane,
                float(item.right.reading.matching_alternatives(
                    item.match.right_b)[0].blocks_per_period))
        except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
            runtime_refusals[item.identity] = (
                "signature_content_invalid", "signature contents changed or became invalid")
            continue
        if (current_left != item.left_content_sha256
                or current_right != item.right_content_sha256):
            runtime_refusals[item.identity] = (
                "signature_content_invalid", "signature contents changed during request")

    combined_refusals = list(foreign_refusals)
    combined_refusals.extend(ShiftDomainRefusal(identity, reason, detail)
                             for identity, (reason, detail) in refusals_by_identity.items())
    combined_refusals.extend(ShiftDomainRefusal(identity, reason, detail)
                             for identity, (reason, detail) in runtime_refusals.items())
    if combined_refusals:
        # Every C1 identity must have one explicit refusal before an all-or-
        # nothing refused result is published.
        existing = {value.match_identity for value in combined_refusals}
        for identity in identities:
            if identity not in existing:
                combined_refusals.append(ShiftDomainRefusal(
                    identity, "shift_domain_incomplete",
                    "another requested identity prevented a complete result"))
        return _result(
            domains=(), refusals=combined_refusals, budget=budget,
            identities=identities, fingerprints=fingerprints, planned=planned,
            consumed=consumed, status="refused", budget_failure_reason=None)
    if len(completed) != len(identities):
        # This is an internal enumeration discrepancy, so no partial evidence
        # or per-match refusals escape.
        return _result(
            domains=(), refusals=(), budget=budget, identities=identities,
            fingerprints=fingerprints, planned=planned, consumed=consumed,
            status="budget_exhausted", budget_failure_reason="shift_domain_incomplete")
    return _result(
        domains=completed, refusals=(), budget=budget, identities=identities,
        fingerprints=fingerprints, planned=planned, consumed=consumed,
        status="complete", budget_failure_reason=None)


def validate_shift_domain_inputs(
    result: ShiftDomainResult,
    support: plane_axis_support.C1AxisSupport,
    matches: Sequence[plane_graph.PlaneMatch],
    signatures: Sequence[lattice_planes.PlaneSignature],
    *,
    readings: Sequence[lattice_planes.PlaneReading],
    max_shift_cells: int,
) -> bool:
    """Check whether the supplied current inputs still match a saved digest.

    The content hashes identify current signature contents. They cannot recover
    content history that C1 did not retain, so callers use this function to
    detect mutation after a P1-D record was made. ``readings`` must be the same
    complete canonical plane universe used for the match indices.
    """
    if (not isinstance(result, ShiftDomainResult)
            or not isinstance(support, plane_axis_support.C1AxisSupport)
            or support.refusal is not None
            or type(max_shift_cells) is not int or max_shift_cells <= 0
            or not isinstance(matches, Sequence)
            or not isinstance(signatures, Sequence)
            or not isinstance(readings, Sequence)):
        return False
    mapped_by_id: dict[plane_axis_support.C1MatchIdentity,
                       list[plane_axis_support.MappedCellShiftAxis]] = {}
    for axis in support.mapped_cell_shifts:
        mapped_by_id.setdefault(axis.match_identity, []).append(axis)
    identities = tuple(sorted(mapped_by_id, key=_identity_key))
    if identities != result.requested_match_identities:
        return False
    candidates: dict[plane_axis_support.C1MatchIdentity, list[plane_graph.PlaneMatch]] = {}
    for match in matches:
        if not isinstance(match, plane_graph.PlaneMatch):
            return False
        try:
            identity = plane_axis_support.C1MatchIdentity.from_match(match)
        except (TypeError, ValueError, OverflowError):
            return False
        if identity in mapped_by_id:
            candidates.setdefault(identity, []).append(match)
    fingerprints = []
    for identity in identities:
        match_rows = candidates.get(identity, [])
        if len(match_rows) != 1:
            return False
        try:
            item, error, contents = _prepare_item(
                identity, match_rows[0], mapped_by_id[identity], readings, signatures, max_shift_cells)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False
        if error is not None or item is None:
            return False
        fingerprints.append((identity, _request_fingerprint(
            identity, max_shift_cells, match_rows, mapped_by_id[identity],
            contents, None)))
    fingerprints_tuple = tuple(fingerprints)
    return _sha256_json(_input_digest_payload(identities, fingerprints_tuple)) == result.input_sha256
