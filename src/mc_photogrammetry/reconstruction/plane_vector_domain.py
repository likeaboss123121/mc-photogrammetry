"""Bounded, provenance-preserving C2-R coupled lateral-vector domains.

This module indexes every two-axis vector induced by one complete P1-D joint
shift and exact P1-B residue records. It neither ranks candidates nor solves
graph cycles or world coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from mc_photogrammetry.reconstruction import lattice_planes
from mc_photogrammetry.reconstruction import plane_axis_support
from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction import plane_residue
from mc_photogrammetry.reconstruction import plane_shift_domain


_DIGEST_LENGTH = 64
_MAX_DETAIL_LENGTH = 240
_REFUSAL_REASONS = frozenset({
    "c1_status_invalid", "p1d_status_invalid", "requested_identity_missing",
    "requested_identity_ambiguous", "provenance_mismatch", "axis_mapping_invalid",
    "residue_evidence_missing", "residue_evidence_ambiguous",
    "residue_candidate_invalid", "joint_shift_invalid",
    "homography_incompatible", "formula_replay_mismatch",
})
_FORMULA_VERSION = "c2r-lateral-vector-v1"
_INDEX_SCHEMA = "c2r-indexed-vector-domain-v1"
_RECONSTRUCTION_VERSION = "c2r-candidate-reconstruct-v1"
_CANONICAL_ORDERING = "entry_row_major_axis0_left_right_axis1_left_right"
_COMPLETENESS_SCOPE = "declared_inputs_only"
_PHYSICAL_TRUTH_MEMBERSHIP = "unverified"
C2R_VECTOR_DOMAIN_SCHEMA = "mc-photogrammetry-c2r-vector-domain-v2"
C2R_VECTOR_DOMAIN_VERSION = 2


def _strict_int(value: Any, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _finite(value: Any) -> float | None:
    if type(value) not in (int, float):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _valid_digest(value: Any) -> bool:
    return (type(value) is str and len(value) == _DIGEST_LENGTH
            and all(character in "0123456789abcdef" for character in value))


def _json_value(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON values must be finite")
        return value
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    raise ValueError("value is not a deterministic JSON primitive")


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_value(value), sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _identity_record(identity: plane_axis_support.C1MatchIdentity) -> dict[str, Any]:
    return identity.to_record()


def _identity_key(identity: plane_axis_support.C1MatchIdentity) -> tuple:
    return identity.sort_key()


def _digest_or_none(value: Any) -> str | None:
    if _valid_digest(value):
        return value
    return None


def _p1d_domain_source(value: plane_shift_domain.MatchShiftDomain) -> dict[str, Any]:
    return {
        "schema": value.schema,
        "version": value.version,
        "correlation_algorithm_id": value.correlation_algorithm_id,
        "match_identity": _identity_record(value.match_identity),
        "replay_sha256": value.replay_sha256,
        "max_shift_cells": value.max_shift_cells,
        "min_overlap_cells": value.min_overlap_cells,
        "right_grid_shape": list(value.right_grid_shape),
        "left_signature_geometry_sha256": value.left_signature_geometry_sha256,
        "right_signature_geometry_sha256": value.right_signature_geometry_sha256,
        "cell_homography_sha256": value.cell_homography_sha256,
        "left_signature_content_sha256": value.left_signature_content_sha256,
        "right_signature_content_sha256": value.right_signature_content_sha256,
    }


def _residue_source_record(value: plane_residue.PeriodicResidueCandidates) -> dict[str, Any]:
    return value.to_record()


def _residue_source_sha256(value: plane_residue.PeriodicResidueCandidates) -> str:
    return _sha256_json(_residue_source_record(value))


@dataclass(frozen=True, slots=True)
class C2RVectorBudget:
    """Explicit positive whole-request limits. No limit has a default."""

    max_requested_match_identities: int
    max_joint_entries: int
    max_axis_residue_pairs: int
    max_vector_candidates: int
    max_provenance_references: int

    def __post_init__(self) -> None:
        if any(not _strict_int(value, minimum=1) for value in (
                self.max_requested_match_identities, self.max_joint_entries,
                self.max_axis_residue_pairs, self.max_vector_candidates,
                self.max_provenance_references)):
            raise ValueError("C2-R vector budget limits must be positive exact integers")

    def to_record(self) -> dict[str, int]:
        return {
            "max_requested_match_identities": self.max_requested_match_identities,
            "max_joint_entries": self.max_joint_entries,
            "max_axis_residue_pairs": self.max_axis_residue_pairs,
            "max_vector_candidates": self.max_vector_candidates,
            "max_provenance_references": self.max_provenance_references,
        }


@dataclass(frozen=True, slots=True)
class C2RVectorWork:
    """Machine-checked planned or represented request-wide work."""

    requested_match_identities: int
    joint_entries: int
    axis_residue_pairs: int
    vector_candidates: int
    provenance_references: int

    def __post_init__(self) -> None:
        if any(not _strict_int(value) for value in (
                self.requested_match_identities, self.joint_entries,
                self.axis_residue_pairs, self.vector_candidates,
                self.provenance_references)):
            raise ValueError("C2-R vector work counters must be non-negative integers")

    def to_record(self) -> dict[str, int]:
        return {
            "requested_match_identities": self.requested_match_identities,
            "joint_entries": self.joint_entries,
            "axis_residue_pairs": self.axis_residue_pairs,
            "vector_candidates": self.vector_candidates,
            "provenance_references": self.provenance_references,
        }


@dataclass(frozen=True, slots=True)
class C2RMaterializationBudget:
    """Explicit caps for an operation that collects full candidate records."""

    max_materialized_candidates: int
    max_materialized_domains: int

    def __post_init__(self) -> None:
        if (not _strict_int(self.max_materialized_candidates, minimum=1)
                or not _strict_int(self.max_materialized_domains, minimum=1)):
            raise ValueError("materialization limits must be positive exact integers")


class C2RMaterializationBudgetExceeded(ValueError):
    """The requested eager tuple exceeds its explicit materialization caps."""


@dataclass(frozen=True, slots=True)
class C2RVectorStorage:
    stored_p1d_entries: int
    stored_residue_values: int
    stored_factor_records: int
    candidates_materialized_during_construction: int

    def __post_init__(self) -> None:
        if any(not _strict_int(value) for value in (
                self.stored_p1d_entries, self.stored_residue_values,
                self.stored_factor_records,
                self.candidates_materialized_during_construction)):
            raise ValueError("indexed-domain storage counters must be non-negative integers")

    def to_record(self) -> dict[str, int]:
        return {
            "stored_p1d_entries": self.stored_p1d_entries,
            "stored_residue_values": self.stored_residue_values,
            "stored_factor_records": self.stored_factor_records,
            "candidates_materialized_during_construction":
                self.candidates_materialized_during_construction,
        }


@dataclass(frozen=True, slots=True)
class ResidueCandidateIdentity:
    """Exact P1-B residue-record identity and one indexed candidate value."""

    frame: str
    plane_index: int
    reading_axis: int
    local_axis: str
    blocks_per_period: float
    source_region_id: str | None
    record_sha256: str
    candidate_index: int
    candidate_value: float

    def __post_init__(self) -> None:
        if (type(self.frame) is not str or not self.frame
                or not _strict_int(self.plane_index)
                or type(self.reading_axis) is not int or self.reading_axis not in (0, 1, 2)
                or self.local_axis not in ("u", "v")
                or _finite(self.blocks_per_period) is None
                or self.blocks_per_period <= 0.0
                or (self.source_region_id is not None
                    and (type(self.source_region_id) is not str
                         or not self.source_region_id))
                or not _valid_digest(self.record_sha256)
                or not _strict_int(self.candidate_index)
                or _finite(self.candidate_value) is None
                or not 0.0 <= self.candidate_value < 1.0):
            raise ValueError("P1-B residue candidate identity is invalid")

    @classmethod
    def from_record(
        cls, record: plane_residue.PeriodicResidueCandidates, index: int,
    ) -> "ResidueCandidateIdentity":
        if (not isinstance(record, plane_residue.PeriodicResidueCandidates)
                or record.refusal is not None
                or type(index) is not int or not 0 <= index < len(record.residue_candidates)):
            raise ValueError("P1-B residue source or candidate index is invalid")
        return cls(
            record.frame, record.plane_index, record.reading_axis,
            record.local_axis, float(record.blocks_per_period),
            record.source_region_id, _residue_source_sha256(record), index,
            float(record.residue_candidates[index]))

    def to_record(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "plane_index": self.plane_index,
            "reading_axis": self.reading_axis,
            "local_axis": self.local_axis,
            "blocks_per_period": float(self.blocks_per_period),
            "source_region_id": self.source_region_id,
            "record_sha256": self.record_sha256,
            "candidate_index": self.candidate_index,
            "candidate_value": float(self.candidate_value),
        }


def _residue_side_payload(value: "ResidueSideTable") -> dict[str, Any]:
    return {
        "endpoint_role": value.endpoint_role,
        "frame": value.frame,
        "plane_index": value.plane_index,
        "reading_axis": value.reading_axis,
        "local_axis": value.local_axis,
        "local_axis_label": value.local_axis_label,
        "blocks_per_period": float(value.blocks_per_period),
        "source_region_id": value.source_region_id,
        "record_sha256": value.record_sha256,
        "candidate_values": [float(item) for item in value.candidate_values],
        "candidate_count": value.candidate_count,
    }


@dataclass(frozen=True, slots=True)
class ResidueSideTable:
    """One compact endpoint table from the exact complete P1-B record."""

    endpoint_role: str
    frame: str
    plane_index: int
    reading_axis: int
    local_axis: str
    local_axis_label: int
    blocks_per_period: float
    source_region_id: str | None
    record_sha256: str
    candidate_values: tuple[float, ...]
    candidate_count: int
    table_sha256: str

    def __post_init__(self) -> None:
        if (self.endpoint_role not in {"left", "right"}
                or type(self.frame) is not str or not self.frame
                or not _strict_int(self.plane_index)
                or type(self.reading_axis) is not int or self.reading_axis not in (0, 1, 2)
                or self.local_axis not in ("u", "v")
                or type(self.local_axis_label) is not int
                or self.local_axis_label not in (0, 1, 2)
                or _finite(self.blocks_per_period) is None
                or self.blocks_per_period <= 0.0
                or (self.source_region_id is not None
                    and (type(self.source_region_id) is not str
                         or not self.source_region_id))
                or not _valid_digest(self.record_sha256)
                or not isinstance(self.candidate_values, tuple)
                or not self.candidate_values
                or any(_finite(value) is None or not 0.0 <= value < 1.0
                       for value in self.candidate_values)
                or any(left >= right for left, right in zip(
                    self.candidate_values, self.candidate_values[1:]))
                or type(self.candidate_count) is not int
                or self.candidate_count != len(self.candidate_values)):
            raise ValueError("P1-B residue side table fields are invalid")
        if not _valid_digest(self.table_sha256):
            raise ValueError("P1-B residue side table digest is invalid")
        if _sha256_json(_residue_side_payload(self)) != self.table_sha256:
            raise ValueError("P1-B residue side table digest does not bind its values")

    @classmethod
    def from_record(
        cls,
        record: plane_residue.PeriodicResidueCandidates,
        *, endpoint_role: str, local_axis_label: int,
    ) -> "ResidueSideTable":
        if (not isinstance(record, plane_residue.PeriodicResidueCandidates)
                or record.refusal is not None):
            raise ValueError("residue side table requires accepted P1-B evidence")
        values = tuple(float(value) for value in record.residue_candidates)
        provisional = object.__new__(cls)
        fields = {
            "endpoint_role": endpoint_role, "frame": record.frame,
            "plane_index": record.plane_index, "reading_axis": record.reading_axis,
            "local_axis": record.local_axis, "local_axis_label": local_axis_label,
            "blocks_per_period": float(record.blocks_per_period),
            "source_region_id": record.source_region_id,
            "record_sha256": _residue_source_sha256(record),
            "candidate_values": values, "candidate_count": len(values),
        }
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "table_sha256",
                           _sha256_json(_residue_side_payload(provisional)))
        cls.__post_init__(provisional)
        return provisional

    def candidate_identity(self, index: int) -> ResidueCandidateIdentity:
        if type(index) is not int or not 0 <= index < self.candidate_count:
            raise IndexError("residue candidate index is outside its side table")
        return ResidueCandidateIdentity(
            self.frame, self.plane_index, self.reading_axis, self.local_axis,
            float(self.blocks_per_period), self.source_region_id,
            self.record_sha256, index, self.candidate_values[index])

    def to_record(self) -> dict[str, Any]:
        record = _residue_side_payload(self)
        record["table_sha256"] = self.table_sha256
        return record


def _axis_factor_payload(value: "LateralAxisFactor") -> dict[str, Any]:
    return {
        "match_identity": _identity_record(value.match_identity),
        "normal_axis": value.normal_axis,
        "world_axis": value.world_axis,
        "left_world_sign": value.left_world_sign,
        "right_world_sign": value.right_world_sign,
        "left_frame_quadrant": value.left_frame_quadrant,
        "right_frame_quadrant": value.right_frame_quadrant,
        "left_local_axis": value.left_local_axis,
        "left_local_axis_label": value.left_local_axis_label,
        "right_local_axis": value.right_local_axis,
        "right_local_axis_label": value.right_local_axis_label,
        "right_shift_source_component": value.right_shift_source_component,
        "left_side": value.left_side.to_record(),
        "right_side": value.right_side.to_record(),
        "c1_axis_source_sha256": value.c1_axis_source_sha256,
        "residue_pair_count": value.residue_pair_count,
    }


@dataclass(frozen=True, slots=True)
class LateralAxisFactor:
    """World-axis mapping and the complete left/right residue product."""

    match_identity: plane_axis_support.C1MatchIdentity
    normal_axis: int
    world_axis: int
    left_world_sign: int
    right_world_sign: int
    left_frame_quadrant: int
    right_frame_quadrant: int
    left_local_axis: str
    left_local_axis_label: int
    right_local_axis: str
    right_local_axis_label: int
    right_shift_source_component: str
    left_side: ResidueSideTable
    right_side: ResidueSideTable
    c1_axis_source_sha256: str
    residue_pair_count: int
    factor_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.match_identity, plane_axis_support.C1MatchIdentity):
            raise ValueError("axis factor requires a selected C1 identity")
        if (type(self.normal_axis) is not int or self.normal_axis not in (0, 1, 2)
                or type(self.world_axis) is not int or self.world_axis not in (0, 1, 2)
                or self.world_axis == self.normal_axis
                or type(self.left_world_sign) is not int or self.left_world_sign not in (-1, 1)
                or type(self.right_world_sign) is not int or self.right_world_sign not in (-1, 1)
                or type(self.left_frame_quadrant) is not int
                or self.left_frame_quadrant not in (0, 1, 2, 3)
                or type(self.right_frame_quadrant) is not int
                or self.right_frame_quadrant not in (0, 1, 2, 3)
                or (self.right_frame_quadrant - self.left_frame_quadrant) % 4
                != self.match_identity.quadrant_of_right):
            raise ValueError("axis factor transform identity is invalid")
        left_tangents = tuple(label for label in range(3)
                              if label != self.match_identity.left_axis)
        right_tangents = tuple(label for label in range(3)
                               if label != self.match_identity.right_axis)
        if (self.left_local_axis not in ("u", "v")
                or self.right_local_axis not in ("u", "v")
                or self.left_local_axis_label != left_tangents[
                    0 if self.left_local_axis == "u" else 1]
                or self.right_local_axis_label != right_tangents[
                    0 if self.right_local_axis == "u" else 1]
                or plane_graph.axis_transform(self.left_local_axis_label,
                                              self.left_frame_quadrant)
                    != (self.world_axis, self.left_world_sign)
                or plane_graph.axis_transform(self.right_local_axis_label,
                                              self.right_frame_quadrant)
                    != (self.world_axis, self.right_world_sign)
                or self.right_shift_source_component
                    != ("column" if self.right_local_axis == "u" else "row")):
            raise ValueError("axis factor local labels/signs do not replay")
        if (not isinstance(self.left_side, ResidueSideTable)
                or not isinstance(self.right_side, ResidueSideTable)
                or self.left_side.endpoint_role != "left"
                or self.right_side.endpoint_role != "right"
                or self.left_side.frame != self.match_identity.left_frame
                or self.right_side.frame != self.match_identity.right_frame
                or self.left_side.plane_index != self.match_identity.left_plane
                or self.right_side.plane_index != self.match_identity.right_plane
                or self.left_side.reading_axis != self.match_identity.left_axis
                or self.right_side.reading_axis != self.match_identity.right_axis
                or self.left_side.local_axis != self.left_local_axis
                or self.right_side.local_axis != self.right_local_axis
                or self.left_side.local_axis_label != self.left_local_axis_label
                or self.right_side.local_axis_label != self.right_local_axis_label):
            raise ValueError("axis factor residue tables differ from its endpoints")
        if (not _valid_digest(self.c1_axis_source_sha256)
                or type(self.residue_pair_count) is not int
                or self.residue_pair_count != self.left_side.candidate_count
                    * self.right_side.candidate_count
                or not _valid_digest(self.factor_sha256)):
            raise ValueError("axis factor source/count/digest is invalid")
        if _sha256_json(_axis_factor_payload(self)) != self.factor_sha256:
            raise ValueError("axis factor digest does not bind its complete product")

    def to_record(self) -> dict[str, Any]:
        record = _axis_factor_payload(self)
        record["factor_sha256"] = self.factor_sha256
        return record


def _axis_term_payload(value: "LateralAxisTerm") -> dict[str, Any]:
    return {
        "formula_version": _FORMULA_VERSION,
        "match_identity": _identity_record(value.match_identity),
        "world_axis": value.world_axis,
        "left_world_sign": value.left_world_sign,
        "right_world_sign": value.right_world_sign,
        "left_frame_quadrant": value.left_frame_quadrant,
        "right_frame_quadrant": value.right_frame_quadrant,
        "left_local_axis": value.left_local_axis,
        "left_local_axis_label": value.left_local_axis_label,
        "right_local_axis": value.right_local_axis,
        "right_local_axis_label": value.right_local_axis_label,
        "shift_entry_rc": list(value.shift_entry_rc),
        "shift_source_component": value.shift_source_component,
        "right_grid_integer_component": value.right_grid_integer_component,
        "signed_shift_component": value.signed_shift_component,
        "left_residue": value.left_residue.to_record(),
        "right_residue": value.right_residue.to_record(),
        "delta": float(value.delta),
    }


@dataclass(frozen=True, slots=True)
class LateralAxisTerm:
    """One mapped-world-axis formula term for a single joint P1-D entry."""

    match_identity: plane_axis_support.C1MatchIdentity
    world_axis: int
    left_world_sign: int
    right_world_sign: int
    left_frame_quadrant: int
    right_frame_quadrant: int
    left_local_axis: str
    left_local_axis_label: int
    right_local_axis: str
    right_local_axis_label: int
    shift_entry_rc: tuple[int, int]
    shift_source_component: str
    right_grid_integer_component: int
    signed_shift_component: int
    left_residue: ResidueCandidateIdentity
    right_residue: ResidueCandidateIdentity
    delta: float
    formula_replay_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.match_identity, plane_axis_support.C1MatchIdentity):
            raise ValueError("lateral axis term requires a C1 match identity")
        if (type(self.world_axis) is not int or self.world_axis not in (0, 1, 2)
                or type(self.left_world_sign) is not int or self.left_world_sign not in (-1, 1)
                or type(self.right_world_sign) is not int or self.right_world_sign not in (-1, 1)
                or type(self.left_frame_quadrant) is not int
                or self.left_frame_quadrant not in (0, 1, 2, 3)
                or type(self.right_frame_quadrant) is not int
                or self.right_frame_quadrant not in (0, 1, 2, 3)
                or (self.right_frame_quadrant - self.left_frame_quadrant) % 4
                != self.match_identity.quadrant_of_right):
            raise ValueError("lateral axis transform identity is invalid")
        if (self.left_local_axis not in ("u", "v")
                or self.right_local_axis not in ("u", "v")
                or type(self.left_local_axis_label) is not int
                or type(self.right_local_axis_label) is not int):
            raise ValueError("lateral local-axis fields are invalid")
        left_tangents = tuple(label for label in range(3)
                              if label != self.match_identity.left_axis)
        right_tangents = tuple(label for label in range(3)
                               if label != self.match_identity.right_axis)
        if (self.left_local_axis_label != left_tangents[
                0 if self.left_local_axis == "u" else 1]
                or self.right_local_axis_label != right_tangents[
                    0 if self.right_local_axis == "u" else 1]):
            raise ValueError("local-axis names disagree with sorted tangent labels")
        if (plane_graph.axis_transform(self.left_local_axis_label,
                                       self.left_frame_quadrant)
                != (self.world_axis, self.left_world_sign)
                or plane_graph.axis_transform(self.right_local_axis_label,
                                              self.right_frame_quadrant)
                != (self.world_axis, self.right_world_sign)):
            raise ValueError("lateral signs do not map both endpoints to the same world axis")
        if (not isinstance(self.shift_entry_rc, tuple) or len(self.shift_entry_rc) != 2
                or any(type(value) is not int for value in self.shift_entry_rc)):
            raise ValueError("lateral term must bind one exact joint row/column entry")
        expected_source = "column" if self.right_local_axis == "u" else "row"
        expected_component = (self.shift_entry_rc[1] if expected_source == "column"
                              else self.shift_entry_rc[0])
        if (self.shift_source_component != expected_source
                or type(self.right_grid_integer_component) is not int
                or self.right_grid_integer_component != expected_component
                or type(self.signed_shift_component) is not int
                or self.signed_shift_component
                != self.right_world_sign * expected_component):
            raise ValueError("lateral shift source must preserve right column=u/row=v")
        if (not isinstance(self.left_residue, ResidueCandidateIdentity)
                or not isinstance(self.right_residue, ResidueCandidateIdentity)):
            raise ValueError("lateral term requires exact left and right P1-B residue identities")
        if (self.left_residue.frame != self.match_identity.left_frame
                or self.left_residue.plane_index != self.match_identity.left_plane
                or self.left_residue.reading_axis != self.match_identity.left_axis
                or self.left_residue.local_axis != self.left_local_axis
                or self.right_residue.frame != self.match_identity.right_frame
                or self.right_residue.plane_index != self.match_identity.right_plane
                or self.right_residue.reading_axis != self.match_identity.right_axis
                or self.right_residue.local_axis != self.right_local_axis):
            raise ValueError("residue source identity does not match the mapped endpoints")
        expected_delta = (
            self.right_world_sign * expected_component
            + self.left_world_sign * self.left_residue.candidate_value
            - self.right_world_sign * self.right_residue.candidate_value)
        if _finite(self.delta) is None or self.delta != expected_delta:
            raise ValueError("lateral delta does not replay the coupled-axis formula")
        if not _valid_digest(self.formula_replay_sha256):
            raise ValueError("lateral formula replay must use lowercase SHA-256")
        if _sha256_json(_axis_term_payload(self)) != self.formula_replay_sha256:
            raise ValueError("lateral formula replay digest does not bind its inputs")

    def to_record(self) -> dict[str, Any]:
        record = _axis_term_payload(self)
        record["formula_replay_sha256"] = self.formula_replay_sha256
        return record


def _candidate_payload(value: "LateralVectorCandidate") -> dict[str, Any]:
    return {
        "match_identity": _identity_record(value.match_identity),
        "p1d_domain_identity": _identity_record(value.p1d_domain_identity),
        "p1d_replay_sha256": value.p1d_replay_sha256,
        "p1d_max_shift_cells": value.p1d_max_shift_cells,
        "p1d_left_signature_content_sha256": value.p1d_left_signature_content_sha256,
        "p1d_right_signature_content_sha256": value.p1d_right_signature_content_sha256,
        "shift_entry": value.shift_entry.to_record(),
        "c1_axis_source_sha256": list(value.c1_axis_source_sha256),
        "left_signature_geometry_sha256": value.left_signature_geometry_sha256,
        "right_signature_geometry_sha256": value.right_signature_geometry_sha256,
        "cell_homography_sha256": value.cell_homography_sha256,
        "axis_terms": [term.to_record() for term in value.axis_terms],
    }


@dataclass(frozen=True, slots=True)
class LateralVectorCandidate:
    """One coupled vector retaining one joint P1-D entry and four residues."""

    match_identity: plane_axis_support.C1MatchIdentity
    p1d_domain_identity: plane_axis_support.C1MatchIdentity
    p1d_replay_sha256: str
    p1d_max_shift_cells: int
    p1d_left_signature_content_sha256: str
    p1d_right_signature_content_sha256: str
    shift_entry: plane_shift_domain.ShiftDomainEntry
    c1_axis_source_sha256: tuple[str, str]
    left_signature_geometry_sha256: str
    right_signature_geometry_sha256: str
    cell_homography_sha256: str
    axis_terms: tuple[LateralAxisTerm, LateralAxisTerm]
    candidate_sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(self.match_identity, plane_axis_support.C1MatchIdentity)
                or self.p1d_domain_identity != self.match_identity
                or type(self.p1d_max_shift_cells) is not int
                or self.p1d_max_shift_cells <= 0
                or not isinstance(self.shift_entry, plane_shift_domain.ShiftDomainEntry)
                or abs(self.shift_entry.row_shift) > self.p1d_max_shift_cells
                or abs(self.shift_entry.column_shift) > self.p1d_max_shift_cells):
            raise ValueError("vector candidate P1-D joint-entry identity/radius is invalid")
        digests = (
            self.p1d_replay_sha256, self.p1d_left_signature_content_sha256,
            self.p1d_right_signature_content_sha256,
            self.left_signature_geometry_sha256, self.right_signature_geometry_sha256,
            self.cell_homography_sha256,
        )
        if not all(_valid_digest(value) for value in digests):
            raise ValueError("vector candidate provenance contains an invalid digest")
        if (not isinstance(self.c1_axis_source_sha256, tuple)
                or len(self.c1_axis_source_sha256) != 2
                or not all(_valid_digest(value) for value in self.c1_axis_source_sha256)):
            raise ValueError("candidate must retain exactly two C1 mapped-axis sources")
        if (not isinstance(self.axis_terms, tuple) or len(self.axis_terms) != 2
                or any(not isinstance(term, LateralAxisTerm) for term in self.axis_terms)
                or tuple(sorted(self.axis_terms, key=lambda term: term.world_axis))
                != self.axis_terms
                or self.axis_terms[0].world_axis == self.axis_terms[1].world_axis
                or any(term.match_identity != self.match_identity
                       or term.shift_entry_rc != (
                           self.shift_entry.row_shift, self.shift_entry.column_shift)
                       for term in self.axis_terms)):
            raise ValueError("candidate axes must be distinct, sorted and share one joint entry")
        if not _valid_digest(self.candidate_sha256):
            raise ValueError("candidate digest must be lowercase SHA-256")
        if _sha256_json(_candidate_payload(self)) != self.candidate_sha256:
            raise ValueError("candidate digest does not bind its contents")

    @property
    def shift_entry_rc(self) -> tuple[int, int]:
        return self.shift_entry.row_shift, self.shift_entry.column_shift

    @property
    def vector(self) -> tuple[float, float]:
        return self.axis_terms[0].delta, self.axis_terms[1].delta

    @property
    def provenance_key(self) -> tuple[Any, ...]:
        return (self.shift_entry_rc,
                tuple((term.world_axis,
                       term.left_residue.record_sha256,
                       term.left_residue.candidate_index,
                       term.right_residue.record_sha256,
                       term.right_residue.candidate_index)
                      for term in self.axis_terms))

    def to_record(self) -> dict[str, Any]:
        record = _candidate_payload(self)
        record["candidate_sha256"] = self.candidate_sha256
        record["shift_entry_rc"] = list(self.shift_entry_rc)
        record["vector"] = list(self.vector)
        record["provenance_key"] = _json_value(self.provenance_key)
        return record


def _finite_compatible_map(
    homography: np.ndarray,
    physical_permutation: np.ndarray,
    left_origin: Sequence[float],
    right_origin: Sequence[float],
    tolerance: float,
    binding: Mapping[str, Any],
) -> str | None:
    """Finite P1-F v2 predicate with internally replayed C1/P1-D fields."""
    if (type(tolerance) not in (int, float) or not math.isfinite(float(tolerance))
            or tolerance < 0.0):
        return "invalid_tolerance"
    value = np.asarray(homography, dtype=np.float64)
    permutation = np.asarray(physical_permutation, dtype=np.float64)
    if value.shape != (3, 3) or permutation.shape != (2, 2):
        return "homography_shape_invalid"
    if not np.isfinite(value).all() or not np.isfinite(permutation).all():
        return "homography_nonfinite"
    denominator = float(value[2, 2])
    if not math.isfinite(denominator) or denominator == 0.0:
        return "homography_denominator_invalid"
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        normalized = value / denominator
    if not np.isfinite(normalized).all():
        return "homography_denominator_unstable"
    if not np.allclose(normalized[2], (0.0, 0.0, 1.0), rtol=0.0, atol=tolerance):
        return "homography_projective"
    if (not np.isin(permutation, (-1.0, 0.0, 1.0)).all()
            or not np.array_equal(np.count_nonzero(permutation, axis=0), (1, 1))
            or not np.array_equal(np.count_nonzero(permutation, axis=1), (1, 1))
            or not np.allclose(normalized[:2, :2], permutation,
                               rtol=0.0, atol=tolerance)):
        return "grid_basis_incompatible"
    left = np.asarray(left_origin, dtype=np.float64)
    right = np.asarray(right_origin, dtype=np.float64)
    if (left.shape != (2,) or right.shape != (2,)
            or not np.isfinite(left).all() or not np.isfinite(right).all()):
        return "grid_origin_incompatible"
    if not np.allclose(normalized[:2, 2], permutation @ right - left,
                       rtol=0.0, atol=tolerance):
        return "grid_origin_incompatible"
    required = {
        "mapping_version", "replayed_homography_sha256", "c1_geometry_digests",
        "recomputed_c1_geometry_digests", "c1_homography_digests",
        "p1d_geometry_digests", "recomputed_p1d_geometry_digests",
        "p1d_content_digests", "recomputed_p1d_content_digests",
        "p1d_homography_digest",
    }
    if not isinstance(binding, Mapping) or not required <= binding.keys():
        return "compatibility_binding_missing"
    tuple_fields = (
        "c1_geometry_digests", "recomputed_c1_geometry_digests",
        "c1_homography_digests", "p1d_geometry_digests",
        "recomputed_p1d_geometry_digests", "p1d_content_digests",
        "recomputed_p1d_content_digests",
    )
    if (any(type(binding[name]) is not tuple for name in tuple_fields)
            or any(len(binding[name]) != 2 for name in (
                "c1_geometry_digests", "recomputed_c1_geometry_digests",
                "p1d_geometry_digests", "recomputed_p1d_geometry_digests",
                "p1d_content_digests", "recomputed_p1d_content_digests"))
            or len(binding["c1_homography_digests"]) != 2):
        return "compatibility_binding_invalid"
    if binding["mapping_version"] != "block-cell-grid-v2":
        return "homography_version_unsupported"
    replayed = binding["replayed_homography_sha256"]
    if not _valid_digest(replayed):
        return "p1f_homography_digest_invalid"
    if binding["c1_geometry_digests"] != binding["recomputed_c1_geometry_digests"]:
        return "c1_geometry_digest_mismatch"
    if any(value != replayed for value in binding["c1_homography_digests"]):
        return "c1_homography_digest_mismatch"
    if binding["p1d_geometry_digests"] != binding["recomputed_p1d_geometry_digests"]:
        return "p1d_geometry_digest_mismatch"
    if binding["p1d_content_digests"] != binding["recomputed_p1d_content_digests"]:
        return "p1d_content_digest_mismatch"
    if binding["p1d_homography_digest"] != replayed:
        return "p1d_homography_digest_mismatch"
    all_digests = [value for name in tuple_fields for value in binding[name]]
    all_digests.extend((binding["p1d_homography_digest"], replayed))
    if any(not _valid_digest(value) for value in all_digests):
        return "compatibility_binding_invalid"
    return None


def _compatibility_payload(
    homography: np.ndarray, permutation: np.ndarray,
    left_origin: Sequence[float], right_origin: Sequence[float],
    tolerance: float, binding: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "homography": np.asarray(homography, dtype=np.float64).tolist(),
        "physical_permutation": np.asarray(permutation, dtype=np.float64).tolist(),
        "left_origin_uv": list(left_origin),
        "right_origin_uv": list(right_origin),
        "comparison_tolerance": float(tolerance),
        "binding": _json_value(binding),
    }


def _mapped_axis_source_digest(value: plane_axis_support.MappedCellShiftAxis) -> str:
    return _sha256_json(value.to_record())


def _normal_constraint_digest(value: plane_axis_support.NormalFaceConstraint) -> str:
    return _sha256_json(value.to_record())


def _domain_payload(value: "MatchLateralVectorDomain") -> dict[str, Any]:
    return {
        "match_identity": _identity_record(value.match_identity),
        "normal_constraint": value.normal_constraint.to_record(),
        "p1d_domain": value.p1d_domain.to_record(),
        "p1b_evidence": value.phase_evidence.to_record(),
        "mapped_axis_sources": [item.to_record() for item in value.c1_mapped_axes],
        "c1_support_sha256": value.c1_support_sha256,
        "p1d_input_sha256": value.p1d_input_sha256,
        "p1b_evidence_sha256": value.p1b_evidence_sha256,
        "canonical_reading_universe_sha256": value.canonical_reading_universe_sha256,
        "left_signature_geometry_sha256": value.left_signature_geometry_sha256,
        "right_signature_geometry_sha256": value.right_signature_geometry_sha256,
        "cell_homography_sha256": value.cell_homography_sha256,
        "left_signature_content_sha256": value.left_signature_content_sha256,
        "right_signature_content_sha256": value.right_signature_content_sha256,
        "comparison_tolerance": float(value.comparison_tolerance),
        "homography": list(value.homography),
        "physical_permutation": list(value.physical_permutation),
        "left_origin_uv": list(value.left_origin_uv),
        "right_origin_uv": list(value.right_origin_uv),
        "compatibility_sha256": value.compatibility_sha256,
        "axis_factors": [factor.to_record() for factor in value.axis_factors],
        "joint_entry_count": value.joint_entry_count,
        "axis_residue_pair_counts": [list(item) for item in value.axis_residue_pair_counts],
        "represented_candidate_count": value.represented_candidate_count,
        "logical_provenance_reference_count": value.logical_provenance_reference_count,
        "index_schema": value.index_schema,
        "reconstruction_version": value.reconstruction_version,
        "ordering_identifier": value.ordering_identifier,
        "shared_provenance_sha256": value.shared_provenance_sha256,
        "candidate_product_sha256": value.candidate_product_sha256,
        "materialization_scope": value.materialization_scope,
        "completeness_scope": value.completeness_scope,
        "physical_truth_membership": value.physical_truth_membership,
        "conditional_truth_predicates": value.conditional_truth_predicates,
    }


def _shared_provenance_payload(value: "MatchLateralVectorDomain") -> dict[str, Any]:
    return {
        "match_identity": _identity_record(value.match_identity),
        "normal_constraint": value.normal_constraint.to_record(),
        "c1_mapped_axes": [item.to_record() for item in value.c1_mapped_axes],
        "c1_support_sha256": value.c1_support_sha256,
        "p1d_input_sha256": value.p1d_input_sha256,
        "p1d_replay_sha256": value.p1d_domain.replay_sha256,
        "p1d_domain_identity": _identity_record(value.p1d_domain.match_identity),
        "p1b_evidence_sha256": value.p1b_evidence_sha256,
        "canonical_reading_universe_sha256": value.canonical_reading_universe_sha256,
        "left_signature_geometry_sha256": value.left_signature_geometry_sha256,
        "right_signature_geometry_sha256": value.right_signature_geometry_sha256,
        "cell_homography_sha256": value.cell_homography_sha256,
        "left_signature_content_sha256": value.left_signature_content_sha256,
        "right_signature_content_sha256": value.right_signature_content_sha256,
        "comparison_tolerance": float(value.comparison_tolerance),
        "homography": list(value.homography),
        "physical_permutation": list(value.physical_permutation),
        "left_origin_uv": list(value.left_origin_uv),
        "right_origin_uv": list(value.right_origin_uv),
        "compatibility_sha256": value.compatibility_sha256,
    }


def _candidate_product_payload(value: "MatchLateralVectorDomain") -> dict[str, Any]:
    return {
        "schema": _INDEX_SCHEMA,
        "reconstruction_version": _RECONSTRUCTION_VERSION,
        "formula_version": _FORMULA_VERSION,
        "ordering": _CANONICAL_ORDERING,
        "p1d_replay_sha256": value.p1d_domain.replay_sha256,
        "p1d_entry_count": value.joint_entry_count,
        "axis_factor_sha256": [factor.factor_sha256 for factor in value.axis_factors],
        "dimensions": [
            [factor.left_side.candidate_count, factor.right_side.candidate_count]
            for factor in value.axis_factors],
        "represented_candidate_count": value.represented_candidate_count,
        "logical_provenance_reference_count": value.logical_provenance_reference_count,
        "shared_provenance_sha256": value.shared_provenance_sha256,
    }


@dataclass(frozen=True, slots=True)
class MatchLateralVectorDomain:
    """Complete factorized product for one C1 match, without eager leaves."""

    match_identity: plane_axis_support.C1MatchIdentity
    normal_constraint: plane_axis_support.NormalFaceConstraint
    p1d_domain: plane_shift_domain.MatchShiftDomain
    phase_evidence: plane_residue.HypothesisPhaseShiftEvidence
    c1_mapped_axes: tuple[plane_axis_support.MappedCellShiftAxis,
                          plane_axis_support.MappedCellShiftAxis]
    c1_support_sha256: str
    p1d_input_sha256: str
    p1b_evidence_sha256: str
    canonical_reading_universe_sha256: str
    left_signature_geometry_sha256: str
    right_signature_geometry_sha256: str
    cell_homography_sha256: str
    left_signature_content_sha256: str
    right_signature_content_sha256: str
    comparison_tolerance: float
    homography: tuple[float, ...]
    physical_permutation: tuple[float, ...]
    left_origin_uv: tuple[float, float]
    right_origin_uv: tuple[float, float]
    compatibility_sha256: str
    axis_factors: tuple[LateralAxisFactor, LateralAxisFactor]
    joint_entry_count: int
    axis_residue_pair_counts: tuple[tuple[int, int], tuple[int, int]]
    represented_candidate_count: int
    logical_provenance_reference_count: int
    index_schema: str
    reconstruction_version: str
    ordering_identifier: str
    shared_provenance_sha256: str
    candidate_product_sha256: str
    materialization_scope: str
    completeness_scope: str
    physical_truth_membership: str
    conditional_truth_predicates: tuple[tuple[str, str], ...]
    domain_sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(self.match_identity, plane_axis_support.C1MatchIdentity)
                or not isinstance(self.normal_constraint,
                                   plane_axis_support.NormalFaceConstraint)
                or self.normal_constraint.match_identity != self.match_identity
                or not isinstance(self.p1d_domain, plane_shift_domain.MatchShiftDomain)
                or self.p1d_domain.match_identity != self.match_identity
                or not isinstance(self.phase_evidence,
                                   plane_residue.HypothesisPhaseShiftEvidence)
                or _phase_identity(self.phase_evidence) != self.match_identity):
            raise ValueError("match domain sources do not share one exact identity")
        if (not isinstance(self.c1_mapped_axes, tuple) or len(self.c1_mapped_axes) != 2
                or any(not isinstance(value, plane_axis_support.MappedCellShiftAxis)
                       for value in self.c1_mapped_axes)
                or any(value.match_identity != self.match_identity
                       for value in self.c1_mapped_axes)
                or len({value.world_axis for value in self.c1_mapped_axes}) != 2
                or self.normal_constraint.world_axis in
                   {value.world_axis for value in self.c1_mapped_axes}):
            raise ValueError("match domain must bind exactly two lateral C1 axes")
        if tuple(sorted(self.c1_mapped_axes, key=lambda item: item.world_axis)) != self.c1_mapped_axes:
            raise ValueError("C1 mapped-axis sources must be world-axis ordered")
        digests = (self.c1_support_sha256, self.p1d_input_sha256,
                   self.p1b_evidence_sha256,
                   self.canonical_reading_universe_sha256,
                   self.left_signature_geometry_sha256,
                   self.right_signature_geometry_sha256,
                   self.cell_homography_sha256,
                   self.left_signature_content_sha256,
                   self.right_signature_content_sha256)
        if not all(_valid_digest(value) for value in digests):
            raise ValueError("match domain provenance digest is invalid")
        if (any(value.match_identity != self.match_identity
                for value in self.c1_mapped_axes)
                or tuple(sorted(self.c1_mapped_axes,
                                key=lambda item: item.world_axis)) != self.c1_mapped_axes):
            raise ValueError("match domain mapped C1 sources are not canonical")
        if (self.p1d_domain.left_signature_geometry_sha256
                != self.left_signature_geometry_sha256
                or self.p1d_domain.right_signature_geometry_sha256
                != self.right_signature_geometry_sha256
                or self.p1d_domain.cell_homography_sha256 != self.cell_homography_sha256
                or self.p1d_domain.left_signature_content_sha256
                != self.left_signature_content_sha256
                or self.p1d_domain.right_signature_content_sha256
                != self.right_signature_content_sha256
                or any((axis.left_signature_geometry_sha256
                        != self.left_signature_geometry_sha256
                        or axis.right_signature_geometry_sha256
                        != self.right_signature_geometry_sha256
                        or axis.cell_homography_sha256 != self.cell_homography_sha256)
                       for axis in self.c1_mapped_axes)):
            raise ValueError("C1 and P1-D geometry/content provenance disagree")
        if (type(self.comparison_tolerance) not in (int, float)
                or not math.isfinite(float(self.comparison_tolerance))
                or self.comparison_tolerance < 0.0
                or not _strict_int(self.joint_entry_count, minimum=1)
                or self.joint_entry_count != len(self.p1d_domain.entries)
                or not isinstance(self.axis_residue_pair_counts, tuple)
                or len(self.axis_residue_pair_counts) != 2
                or any(type(item) is not tuple or len(item) != 2
                       or type(item[0]) is not int or type(item[1]) is not int
                       or item[1] < 1 for item in self.axis_residue_pair_counts)
                or tuple(item[0] for item in self.axis_residue_pair_counts)
                != tuple(axis.world_axis for axis in self.c1_mapped_axes)):
            raise ValueError("match domain count or tolerance fields are invalid")
        if (not isinstance(self.axis_factors, tuple) or len(self.axis_factors) != 2
                or any(not isinstance(item, LateralAxisFactor)
                       for item in self.axis_factors)
                or tuple(sorted(self.axis_factors,
                                key=lambda item: item.world_axis)) != self.axis_factors
                or any(item.match_identity != self.match_identity
                       or item.normal_axis != self.normal_constraint.world_axis
                       or item.left_frame_quadrant != self.normal_constraint.left_frame_quadrant
                       or item.right_frame_quadrant != self.normal_constraint.right_frame_quadrant
                       for item in self.axis_factors)
                or tuple((item.world_axis, item.residue_pair_count)
                         for item in self.axis_factors) != self.axis_residue_pair_counts):
            raise ValueError("match domain must bind two exact world-axis residue factors")
        if _sha256_json(self.phase_evidence.to_record()) != self.p1b_evidence_sha256:
            raise ValueError("match domain P1-B digest does not replay from its exact source")
        expected_rows = _axis_rows_for(
            self.match_identity, self.normal_constraint,
            self.c1_mapped_axes, self.phase_evidence)
        for factor, row in zip(self.axis_factors, expected_rows):
            mapped = [axis for axis in self.c1_mapped_axes
                      if axis.world_axis == row.world_axis]
            expected_left = ResidueSideTable.from_record(
                row.left_record, endpoint_role="left",
                local_axis_label=row.left_label)
            expected_right = ResidueSideTable.from_record(
                row.right_record, endpoint_role="right",
                local_axis_label=row.right_label)
            if (len(mapped) != 1
                    or factor.c1_axis_source_sha256
                        != _mapped_axis_source_digest(mapped[0])
                    or factor.left_side != expected_left
                    or factor.right_side != expected_right):
                raise ValueError("axis factor does not replay exact C1/P1-B source content")
        if (not isinstance(self.homography, tuple) or len(self.homography) != 9
                or any(_finite(value) is None for value in self.homography)
                or not isinstance(self.physical_permutation, tuple)
                or len(self.physical_permutation) != 4
                or any(_finite(value) is None for value in self.physical_permutation)
                or not isinstance(self.left_origin_uv, tuple)
                or len(self.left_origin_uv) != 2
                or not isinstance(self.right_origin_uv, tuple)
                or len(self.right_origin_uv) != 2
                or any(_finite(value) is None
                       for value in (*self.left_origin_uv, *self.right_origin_uv))
                or not _valid_digest(self.compatibility_sha256)):
            raise ValueError("match domain finite map provenance is malformed")
        binding = _build_binding(
            self.match_identity, self.c1_mapped_axes, self.p1d_domain,
            self.left_signature_geometry_sha256,
            self.right_signature_geometry_sha256,
            self.left_signature_content_sha256,
            self.right_signature_content_sha256,
            self.cell_homography_sha256)
        homography = np.asarray(self.homography, dtype=np.float64).reshape(3, 3)
        permutation = np.asarray(self.physical_permutation, dtype=np.float64).reshape(2, 2)
        if _finite_compatible_map(
                homography, permutation, self.left_origin_uv, self.right_origin_uv,
                float(self.comparison_tolerance), binding) is not None:
            raise ValueError("match domain finite grid-map compatibility failed")
        if _sha256_json(_compatibility_payload(
                homography, permutation, self.left_origin_uv, self.right_origin_uv,
                float(self.comparison_tolerance), binding)) != self.compatibility_sha256:
            raise ValueError("match domain compatibility digest does not replay")
        expected_count = (self.joint_entry_count
                          * self.axis_factors[0].residue_pair_count
                          * self.axis_factors[1].residue_pair_count)
        if (type(self.represented_candidate_count) is not int
                or self.represented_candidate_count != expected_count
                or type(self.logical_provenance_reference_count) is not int
                or self.logical_provenance_reference_count != 7 * expected_count + 1):
            raise ValueError("indexed product dimensions disagree with represented work")
        if (self.index_schema != _INDEX_SCHEMA
                or self.reconstruction_version != _RECONSTRUCTION_VERSION
                or self.ordering_identifier != _CANONICAL_ORDERING
                or self.materialization_scope != "index_only"):
            raise ValueError("indexed domain schema/order/materialization contract is invalid")
        if (not _valid_digest(self.shared_provenance_sha256)
                or self.shared_provenance_sha256
                    != _sha256_json(_shared_provenance_payload(self))
                or not _valid_digest(self.candidate_product_sha256)
                or self.candidate_product_sha256
                    != _sha256_json(_candidate_product_payload(self))):
            raise ValueError("indexed domain product commitments do not replay")
        if (self.completeness_scope != _COMPLETENESS_SCOPE
                or self.physical_truth_membership != _PHYSICAL_TRUTH_MEMBERSHIP):
            raise ValueError("match domain must state its declared-input and truth limits")
        expected_predicates = (
            ("c1_provenance_replayed", "verified"),
            ("same_physical_plane_lattice", "unverified"),
            ("true_local_residues_in_p1b", "unverified"),
            ("true_integer_carries_in_p1d", "unverified"),
            ("p1f_grid_map_and_correlation_convention", "verified"),
        )
        if self.conditional_truth_predicates != expected_predicates:
            raise ValueError("conditional truth-membership labels are invalid")
        if not _valid_digest(self.domain_sha256):
            raise ValueError("match lateral-vector domain digest is invalid")
        if _sha256_json(_domain_payload(self)) != self.domain_sha256:
            raise ValueError("match lateral-vector domain digest does not bind its contents")

    def to_record(self) -> dict[str, Any]:
        record = _domain_payload(self)
        record["domain_sha256"] = self.domain_sha256
        return record

    @property
    def candidate_count(self) -> int:
        return self.represented_candidate_count

    @property
    def provenance_reference_count(self) -> int:
        return self.logical_provenance_reference_count

    def candidate_at(self, index: int) -> LateralVectorCandidate:
        return _candidate_at_index(self, index)

    def provenance_key_at(self, index: int) -> tuple[Any, ...]:
        entry_index, indices = _decode_mixed_radix(self, index)
        entry = self.p1d_domain.entries[entry_index]
        return _provenance_key_from_parts(self, entry, indices)

    def vector_at(self, index: int) -> tuple[tuple[int, float], tuple[int, float]]:
        entry_index, indices = _decode_mixed_radix(self, index)
        entry = self.p1d_domain.entries[entry_index]
        return _vector_from_parts(self, entry, indices)

    def iter_candidates(self, start: int, stop: int):
        _check_index_range(self, start, stop)
        for index in range(start, stop):
            yield self.candidate_at(index)

    def iter_views(self, start: int, stop: int):
        _check_index_range(self, start, stop)
        for index in range(start, stop):
            yield _view_at_index(self, index)

    def iter_entry_groups(self):
        vectors_per_entry = (self.represented_candidate_count // self.joint_entry_count)
        for index, entry in enumerate(self.p1d_domain.entries):
            start = index * vectors_per_entry
            yield entry, start, start + vectors_per_entry

    def range_sha256(self, start: int, stop: int) -> str:
        _check_index_range(self, start, stop)
        digest = hashlib.sha256()
        for view in self.iter_views(start, stop):
            digest.update(_canonical_json(view.to_record()).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()



@dataclass(frozen=True, slots=True)
class LateralVectorView:
    candidate_index: int
    shift_entry_rc: tuple[int, int]
    provenance_key: tuple[Any, ...]
    vector: tuple[tuple[int, float], tuple[int, float]]

    def __post_init__(self) -> None:
        if (type(self.candidate_index) is not int or self.candidate_index < 0
                or not isinstance(self.shift_entry_rc, tuple)
                or len(self.shift_entry_rc) != 2
                or any(type(value) is not int for value in self.shift_entry_rc)
                or not isinstance(self.vector, tuple) or len(self.vector) != 2
                or any(type(axis) is not int or axis not in (0, 1, 2)
                       or _finite(value) is None for axis, value in self.vector)
                or self.vector[0][0] >= self.vector[1][0]):
            raise ValueError("compact candidate view is invalid")
        _json_value(self.provenance_key)

    def to_record(self) -> dict[str, Any]:
        return {"candidate_index": self.candidate_index,
                "shift_entry_rc": list(self.shift_entry_rc),
                "provenance_key": _json_value(self.provenance_key),
                "vector": [[axis, value] for axis, value in self.vector]}


def _check_index_range(domain: MatchLateralVectorDomain, start: int, stop: int) -> None:
    if (type(start) is not int or type(stop) is not int
            or start < 0 or stop < start or stop > domain.represented_candidate_count):
        raise ValueError("candidate range must be an explicit valid half-open interval")


def _decode_mixed_radix(
    domain: MatchLateralVectorDomain, index: int,
) -> tuple[int, tuple[int, int, int, int]]:
    if type(index) is not int or not 0 <= index < domain.represented_candidate_count:
        raise IndexError("candidate index is outside the represented product")
    factor0, factor1 = domain.axis_factors
    a0, b0 = factor0.left_side.candidate_count, factor0.right_side.candidate_count
    a1, b1 = factor1.left_side.candidate_count, factor1.right_side.candidate_count
    p0, p1 = a0 * b0, a1 * b1
    vectors_per_entry = p0 * p1
    entry_index = index // vectors_per_entry
    tail = index % vectors_per_entry
    q0, q1 = tail // p1, tail % p1
    return entry_index, (q0 // b0, q0 % b0, q1 // b1, q1 % b1)


def _provenance_key_from_parts(
    domain: MatchLateralVectorDomain,
    entry: plane_shift_domain.ShiftDomainEntry,
    indices: tuple[int, int, int, int],
) -> tuple[Any, ...]:
    axes = tuple((factor.world_axis, factor.left_side.record_sha256, left_index,
                  factor.right_side.record_sha256, right_index)
                 for factor, left_index, right_index in (
                     (domain.axis_factors[0], indices[0], indices[1]),
                     (domain.axis_factors[1], indices[2], indices[3])))
    return ((entry.row_shift, entry.column_shift), axes)


def _vector_from_parts(
    domain: MatchLateralVectorDomain,
    entry: plane_shift_domain.ShiftDomainEntry,
    indices: tuple[int, int, int, int],
) -> tuple[tuple[int, float], tuple[int, float]]:
    values = []
    for factor, left_index, right_index in (
            (domain.axis_factors[0], indices[0], indices[1]),
            (domain.axis_factors[1], indices[2], indices[3])):
        component = (entry.column_shift
                     if factor.right_shift_source_component == "column"
                     else entry.row_shift)
        delta = (factor.right_world_sign * component
                 + factor.left_world_sign * factor.left_side.candidate_values[left_index]
                 - factor.right_world_sign * factor.right_side.candidate_values[right_index])
        values.append((factor.world_axis, float(delta)))
    return values[0], values[1]


def _view_at_index(domain: MatchLateralVectorDomain, index: int) -> LateralVectorView:
    entry_index, indices = _decode_mixed_radix(domain, index)
    entry = domain.p1d_domain.entries[entry_index]
    return LateralVectorView(
        index, (entry.row_shift, entry.column_shift),
        _provenance_key_from_parts(domain, entry, indices),
        _vector_from_parts(domain, entry, indices))


def _term_from_factor(
    identity: plane_axis_support.C1MatchIdentity,
    normal: plane_axis_support.NormalFaceConstraint,
    factor: LateralAxisFactor,
    entry: plane_shift_domain.ShiftDomainEntry,
    left_index: int,
    right_index: int,
) -> LateralAxisTerm:
    left_residue = factor.left_side.candidate_identity(left_index)
    right_residue = factor.right_side.candidate_identity(right_index)
    component = (entry.column_shift
                 if factor.right_shift_source_component == "column"
                 else entry.row_shift)
    delta = (factor.right_world_sign * component
             + factor.left_world_sign * left_residue.candidate_value
             - factor.right_world_sign * right_residue.candidate_value)
    provisional = object.__new__(LateralAxisTerm)
    fields = {
        "match_identity": identity, "world_axis": factor.world_axis,
        "left_world_sign": factor.left_world_sign,
        "right_world_sign": factor.right_world_sign,
        "left_frame_quadrant": factor.left_frame_quadrant,
        "right_frame_quadrant": factor.right_frame_quadrant,
        "left_local_axis": factor.left_local_axis,
        "left_local_axis_label": factor.left_local_axis_label,
        "right_local_axis": factor.right_local_axis,
        "right_local_axis_label": factor.right_local_axis_label,
        "shift_entry_rc": (entry.row_shift, entry.column_shift),
        "shift_source_component": factor.right_shift_source_component,
        "right_grid_integer_component": component,
        "signed_shift_component": factor.right_world_sign * component,
        "left_residue": left_residue, "right_residue": right_residue,
        "delta": float(delta),
    }
    for name, value in fields.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "formula_replay_sha256",
                       _sha256_json(_axis_term_payload(provisional)))
    LateralAxisTerm.__post_init__(provisional)
    return provisional


def _candidate_at_index(
    domain: MatchLateralVectorDomain, index: int,
) -> LateralVectorCandidate:
    entry_index, indices = _decode_mixed_radix(domain, index)
    entry = domain.p1d_domain.entries[entry_index]
    terms = (
        _term_from_factor(domain.match_identity, domain.normal_constraint,
                          domain.axis_factors[0], entry, indices[0], indices[1]),
        _term_from_factor(domain.match_identity, domain.normal_constraint,
                          domain.axis_factors[1], entry, indices[2], indices[3]),
    )
    provisional = object.__new__(LateralVectorCandidate)
    fields = {
        "match_identity": domain.match_identity,
        "p1d_domain_identity": domain.p1d_domain.match_identity,
        "p1d_replay_sha256": domain.p1d_domain.replay_sha256,
        "p1d_max_shift_cells": domain.p1d_domain.max_shift_cells,
        "p1d_left_signature_content_sha256":
            domain.p1d_domain.left_signature_content_sha256,
        "p1d_right_signature_content_sha256":
            domain.p1d_domain.right_signature_content_sha256,
        "shift_entry": entry,
        "c1_axis_source_sha256": tuple(
            factor.c1_axis_source_sha256 for factor in domain.axis_factors),
        "left_signature_geometry_sha256": domain.left_signature_geometry_sha256,
        "right_signature_geometry_sha256": domain.right_signature_geometry_sha256,
        "cell_homography_sha256": domain.cell_homography_sha256,
        "axis_terms": terms,
    }
    for name, value in fields.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "candidate_sha256",
                       _sha256_json(_candidate_payload(provisional)))
    LateralVectorCandidate.__post_init__(provisional)
    return provisional


def _storage_for_domains(
    domains: Sequence[MatchLateralVectorDomain],
) -> C2RVectorStorage:
    return C2RVectorStorage(
        sum(domain.joint_entry_count for domain in domains),
        sum(sum(side.candidate_count for factor in domain.axis_factors
                for side in (factor.left_side, factor.right_side))
            for domain in domains),
        6 * len(domains), 0)


def materialize_lateral_vector_domains(
    result: "C2RVectorDomainResult",
    budget: C2RMaterializationBudget,
) -> tuple[tuple[LateralVectorCandidate, ...], ...]:
    """Explicitly collect all indexed candidates after atomic budget preflight."""
    if not isinstance(result, C2RVectorDomainResult):
        raise TypeError("materialization requires a C2RVectorDomainResult")
    if not isinstance(budget, C2RMaterializationBudget):
        raise TypeError("an explicit C2RMaterializationBudget is required")
    result.__post_init__()
    if result.status != "complete":
        raise ValueError("only a complete indexed result can be materialized")
    domain_count = len(result.domains)
    candidate_count = sum(domain.represented_candidate_count
                          for domain in result.domains)
    if (domain_count > budget.max_materialized_domains
            or candidate_count > budget.max_materialized_candidates):
        raise C2RMaterializationBudgetExceeded(
            "complete materialization exceeds its explicit all-or-nothing budget")
    return tuple(tuple(domain.iter_candidates(0, domain.represented_candidate_count))
                 for domain in result.domains)


def _phase_identity(value: plane_residue.HypothesisPhaseShiftEvidence
                    ) -> plane_axis_support.C1MatchIdentity:
    return plane_axis_support.C1MatchIdentity(
        value.left_frame, value.right_frame, value.left_plane, value.right_plane,
        value.left_axis, value.right_axis, float(value.left_b), float(value.right_b),
        value.quadrant_of_right)


def _result_input_payload(value: "C2RVectorDomainResult") -> dict[str, Any]:
    return {
        "schema": value.schema,
        "version": value.version,
        "correlation_algorithm_id": value.correlation_algorithm_id,
        "canonical_reading_universe_sha256": value.canonical_reading_universe_sha256,
        "c1_support_sha256": value.c1_support_sha256,
        "p1d_input_sha256": value.p1d_input_sha256,
        "selected_matches_sha256": value.selected_matches_sha256,
        "phase_evidence_sha256": [
            {"match_identity": _identity_record(identity), "sha256": digest}
            for identity, digest in value.phase_evidence_sha256],
        "comparison_tolerance": value.comparison_tolerance,
        "invalid_tolerance_input": value.invalid_tolerance_input,
        "budget": value.budget.to_record(),
        "requested_match_identities": [
            _identity_record(identity) for identity in value.requested_match_identities],
    }


def _result_payload(value: "C2RVectorDomainResult") -> dict[str, Any]:
    return {
        "input": _result_input_payload(value),
        "requested_match_identities": [
            _identity_record(identity) for identity in value.requested_match_identities],
        "preflight_plans": [plan.to_record() for plan in value._preflight_plans],
        "planned_work": value.planned_work.to_record(),
        "represented_work": value.represented_work.to_record(),
        "storage": _storage_for_domains(value.domains).to_record(),
        "domains": [domain.to_record() for domain in value.domains],
        "refusals": [refusal.to_record() for refusal in value.refusals],
        "status": value.status,
        "budget_failure_reason": value.budget_failure_reason,
        "completeness_scope": value.completeness_scope,
        "physical_truth_membership": value.physical_truth_membership,
    }


@dataclass(frozen=True, slots=True)
class C2RVectorRefusal:
    match_identity: plane_axis_support.C1MatchIdentity | None
    reason: str
    detail: str

    def __post_init__(self) -> None:
        if (self.match_identity is not None and not isinstance(
                self.match_identity, plane_axis_support.C1MatchIdentity)):
            raise ValueError("C2-R refusal identity is invalid")
        if (type(self.reason) is not str or self.reason not in _REFUSAL_REASONS
                or type(self.detail) is not str or len(self.detail) > _MAX_DETAIL_LENGTH
                or any(ord(character) < 32 for character in self.detail)):
            raise ValueError("C2-R refusal fields are invalid")

    def to_record(self) -> dict[str, Any]:
        return {"match_identity": (None if self.match_identity is None
                                   else _identity_record(self.match_identity)),
                "reason": self.reason, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class _C2RMatchPlan:
    match_identity: plane_axis_support.C1MatchIdentity
    joint_entries: int
    axis_pair_counts: tuple[tuple[int, int], ...]
    refusal_reason: str | None
    refusal_detail: str | None

    def __post_init__(self) -> None:
        if (not isinstance(self.match_identity, plane_axis_support.C1MatchIdentity)
                or not _strict_int(self.joint_entries)
                or not isinstance(self.axis_pair_counts, tuple)
                or any(type(item) is not tuple or len(item) != 2
                       or type(item[0]) is not int or item[0] not in (0, 1, 2)
                       or not _strict_int(item[1]) for item in self.axis_pair_counts)
                or len({item[0] for item in self.axis_pair_counts}) != len(
                    self.axis_pair_counts)
                or (self.refusal_reason is not None
                    and self.refusal_reason not in _REFUSAL_REASONS)
                or (self.refusal_reason is None) != (self.refusal_detail is None)):
            raise ValueError("C2-R preflight plan is malformed")
        if self.refusal_reason is None and (
                self.joint_entries <= 0 or len(self.axis_pair_counts) != 2
                or any(count <= 0 for _axis, count in self.axis_pair_counts)):
            raise ValueError("ready preflight requires entries and both residue-pair axes")
        if self.refusal_detail is not None and (
                type(self.refusal_detail) is not str
                or len(self.refusal_detail) > _MAX_DETAIL_LENGTH):
            raise ValueError("C2-R preflight refusal detail is invalid")

    @property
    def axis_residue_pairs(self) -> int:
        return self.joint_entries * sum(count for _axis, count in self.axis_pair_counts)

    @property
    def vector_candidates(self) -> int:
        if len(self.axis_pair_counts) != 2:
            return 0
        return (self.joint_entries * self.axis_pair_counts[0][1]
                * self.axis_pair_counts[1][1])

    def to_record(self) -> dict[str, Any]:
        return {
            "match_identity": _identity_record(self.match_identity),
            "joint_entries": self.joint_entries,
            "axis_pair_counts": [list(value) for value in self.axis_pair_counts],
            "axis_residue_pairs": self.axis_residue_pairs,
            "vector_candidates": self.vector_candidates,
            "provenance_references": 7 * self.vector_candidates + 1,
            "refusal_reason": self.refusal_reason,
            "refusal_detail": self.refusal_detail,
        }


def _work_from_plans(plans: Sequence[_C2RMatchPlan]) -> C2RVectorWork:
    requested = len(plans)
    joint = sum(plan.joint_entries for plan in plans)
    pairs = sum(plan.axis_residue_pairs for plan in plans)
    vectors = sum(plan.vector_candidates for plan in plans)
    return C2RVectorWork(requested, joint, pairs, vectors, 7 * vectors + requested)


@dataclass(frozen=True, slots=True)
class C2RVectorDomainResult:
    schema: str
    version: int
    correlation_algorithm_id: str
    domains: tuple[MatchLateralVectorDomain, ...]
    refusals: tuple[C2RVectorRefusal, ...]
    budget: C2RVectorBudget
    requested_match_identities: tuple[plane_axis_support.C1MatchIdentity, ...]
    _preflight_plans: tuple[_C2RMatchPlan, ...]
    planned_work: C2RVectorWork
    represented_work: C2RVectorWork
    canonical_reading_universe_sha256: str
    c1_support_sha256: str
    p1d_input_sha256: str
    selected_matches_sha256: str
    phase_evidence_sha256: tuple[tuple[plane_axis_support.C1MatchIdentity, str], ...]
    comparison_tolerance: float | None
    invalid_tolerance_input: str | None
    status: str
    budget_failure_reason: str | None
    completeness_scope: str
    physical_truth_membership: str
    input_sha256: str
    result_sha256: str

    def __post_init__(self) -> None:
        if (self.schema != C2R_VECTOR_DOMAIN_SCHEMA
                or type(self.version) is not int
                or self.version != C2R_VECTOR_DOMAIN_VERSION
                or self.correlation_algorithm_id
                != plane_graph.PLANE_CORRELATION_ALGORITHM):
            raise ValueError("C2-R vector-domain provenance is unsupported")
        if not isinstance(self.budget, C2RVectorBudget):
            raise ValueError("C2-R result requires its explicit budget")
        if (not isinstance(self.requested_match_identities, tuple)
                or any(not isinstance(value, plane_axis_support.C1MatchIdentity)
                       for value in self.requested_match_identities)
                or tuple(sorted(set(self.requested_match_identities), key=_identity_key))
                != self.requested_match_identities
                or not isinstance(self._preflight_plans, tuple)
                or any(not isinstance(value, _C2RMatchPlan)
                       for value in self._preflight_plans)
                or tuple(plan.match_identity for plan in self._preflight_plans)
                != self.requested_match_identities):
            raise ValueError("C2-R result request/preflight identities are not canonical")
        if (not isinstance(self.domains, tuple)
                or any(not isinstance(value, MatchLateralVectorDomain)
                       for value in self.domains)
                or not isinstance(self.refusals, tuple)
                or any(not isinstance(value, C2RVectorRefusal)
                       for value in self.refusals)
                or not isinstance(self.planned_work, C2RVectorWork)
                or not isinstance(self.represented_work, C2RVectorWork)):
            raise ValueError("C2-R result records are malformed")
        if self.planned_work != _work_from_plans(self._preflight_plans):
            raise ValueError("planned C2-R work does not replay from internal match plans")
        if (not isinstance(self.phase_evidence_sha256, tuple)
                or tuple(identity for identity, _digest in self.phase_evidence_sha256)
                != self.requested_match_identities
                or any(not _valid_digest(digest)
                       for _identity, digest in self.phase_evidence_sha256)):
            raise ValueError("C2-R P1-B evidence digests are incomplete")
        if not all(_valid_digest(value) for value in (
                self.canonical_reading_universe_sha256, self.c1_support_sha256,
                self.p1d_input_sha256, self.selected_matches_sha256,
                self.input_sha256, self.result_sha256)):
            raise ValueError("C2-R result provenance digest is invalid")
        if ((self.comparison_tolerance is None)
                != (self.invalid_tolerance_input is not None)):
            raise ValueError("C2-R tolerance validity tag is inconsistent")
        if self.comparison_tolerance is not None and (
                type(self.comparison_tolerance) not in (int, float)
                or not math.isfinite(float(self.comparison_tolerance))
                or self.comparison_tolerance < 0.0):
            raise ValueError("C2-R result comparison tolerance is invalid")
        if (self.invalid_tolerance_input is not None
                and (type(self.invalid_tolerance_input) is not str
                     or not self.invalid_tolerance_input)):
            raise ValueError("invalid tolerance input marker is malformed")
        if (self.status not in {"complete", "refused", "budget_exhausted"}
                or self.completeness_scope != _COMPLETENESS_SCOPE
                or self.physical_truth_membership != _PHYSICAL_TRUTH_MEMBERSHIP):
            raise ValueError("C2-R result status or completeness labels are invalid")
        limits = self.budget.to_record()
        work = self.planned_work.to_record()
        over = any(work[key] > limits["max_" + key] for key in work)
        zero = C2RVectorWork(0, 0, 0, 0, 0)
        domain_ids = tuple(value.match_identity for value in self.domains)
        if self.status == "complete":
            if (self.comparison_tolerance is None or over or self.refusals
                    or domain_ids != self.requested_match_identities
                    or not domain_ids or self.budget_failure_reason is not None
                    or self.represented_work != self.planned_work
                    or any(plan.refusal_reason is not None
                           for plan in self._preflight_plans)):
                raise ValueError("complete C2-R result must publish every planned candidate")
            if _work_from_domains(self.domains) != self.planned_work:
                raise ValueError("complete C2-R work differs from retained domain records")
            phase_digests = dict(self.phase_evidence_sha256)
            if any(
                    domain.c1_support_sha256 != self.c1_support_sha256
                    or domain.p1d_input_sha256 != self.p1d_input_sha256
                    or domain.canonical_reading_universe_sha256
                        != self.canonical_reading_universe_sha256
                    or domain.p1b_evidence_sha256
                        != phase_digests.get(domain.match_identity)
                    or domain.comparison_tolerance != self.comparison_tolerance
                    for domain in self.domains):
                raise ValueError("indexed domains differ from request-wide source provenance")
        elif self.status == "refused":
            refusal_ids = tuple(value.match_identity for value in self.refusals
                                if value.match_identity is not None)
            if (self.domains or self.budget_failure_reason is not None or over
                    or self.represented_work != zero or not self.refusals):
                raise ValueError("refused C2-R result must be atomic and contain no candidates")
            if self.requested_match_identities:
                failed_plans = tuple(plan for plan in self._preflight_plans
                                     if plan.refusal_reason is not None)
                expected_ids = tuple(plan.match_identity for plan in failed_plans)
                if (not expected_ids or any(value is None for value in refusal_ids)
                        or len(refusal_ids) != len(set(refusal_ids))
                        or set(refusal_ids) != set(expected_ids)):
                    raise ValueError(
                        "refused C2-R result refusals must match exactly failed selected identities")
                refusal_by_id = {value.match_identity: value for value in self.refusals}
                if any(refusal_by_id[plan.match_identity].reason != plan.refusal_reason
                       or refusal_by_id[plan.match_identity].detail != plan.refusal_detail
                       for plan in failed_plans):
                    raise ValueError("refused C2-R diagnostics differ from their preflight plans")
            elif (len(self.refusals) != 1
                  or self.refusals[0].match_identity is not None
                  or self._preflight_plans):
                raise ValueError("identity-free refusal must contain one global diagnostic")
        else:
            if (self.domains or self.refusals or self.represented_work != zero
                    or self.budget_failure_reason not in {
                        "planned_budget_exceeded", "runtime_count_mismatch"}
                    or (self.budget_failure_reason == "planned_budget_exceeded") != over):
                raise ValueError("budget-exhausted result must discard all candidates and refusals")
        if tuple(sorted(self.refusals, key=_refusal_key)) != self.refusals:
            raise ValueError("C2-R refusals must be deterministically ordered")
        if _sha256_json(_result_input_payload(self)) != self.input_sha256:
            raise ValueError("C2-R input digest does not bind canonical inputs")
        if _sha256_json(_result_payload(self)) != self.result_sha256:
            raise ValueError("C2-R result digest does not bind its records")

    def to_record(self) -> dict[str, Any]:
        record = _result_payload(self)
        record["input_sha256"] = self.input_sha256
        record["result_sha256"] = self.result_sha256
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return record

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.to_record(), sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")



def _refusal_key(value: C2RVectorRefusal) -> tuple[Any, ...]:
    return (() if value.match_identity is None else _identity_key(value.match_identity),
            value.reason, value.detail)


def _work_for_domains(domains: Sequence[MatchLateralVectorDomain]) -> C2RVectorWork:
    requested = len(domains)
    joint = sum(len(domain.p1d_domain.entries) for domain in domains)
    pairs = sum(len(domain.p1d_domain.entries) * sum(
        factor.left_side.candidate_count * factor.right_side.candidate_count
        for factor in domain.axis_factors) for domain in domains)
    vectors = sum(len(domain.p1d_domain.entries)
                  * math.prod(factor.left_side.candidate_count
                              * factor.right_side.candidate_count
                              for factor in domain.axis_factors)
                  for domain in domains)
    return C2RVectorWork(requested, joint, pairs, vectors, 7 * vectors + requested)


def _work_from_domains(domains: Sequence[MatchLateralVectorDomain]) -> C2RVectorWork:
    return _work_for_domains(domains)


@dataclass(frozen=True, slots=True)
class _AxisRow:
    world_axis: int
    left_name: str
    left_label: int
    right_name: str
    right_label: int
    left_sign: int
    right_sign: int
    left_record: plane_residue.PeriodicResidueCandidates
    right_record: plane_residue.PeriodicResidueCandidates


@dataclass(frozen=True, slots=True)
class _PreparedMatch:
    identity: plane_axis_support.C1MatchIdentity
    normal: plane_axis_support.NormalFaceConstraint
    p1d_domain: plane_shift_domain.MatchShiftDomain
    phase: plane_residue.HypothesisPhaseShiftEvidence
    mapped_axes: tuple[plane_axis_support.MappedCellShiftAxis,
                       plane_axis_support.MappedCellShiftAxis]
    left_signature_geometry_sha256: str
    right_signature_geometry_sha256: str
    cell_homography_sha256: str
    left_signature_content_sha256: str
    right_signature_content_sha256: str
    left_signature: lattice_planes.PlaneSignature
    right_signature: lattice_planes.PlaneSignature
    left_canonical_b: float
    right_canonical_b: float
    homography: tuple[float, ...]
    physical_permutation: tuple[float, ...]
    left_origin_uv: tuple[float, float]
    right_origin_uv: tuple[float, float]
    axis_rows: tuple[_AxisRow, _AxisRow]
    axis_pair_counts: tuple[tuple[int, int], tuple[int, int]]
    phase_sha256: str
    compatibility_sha256: str


def _canonical_reading_universe_sha256(
    readings: Sequence[lattice_planes.PlaneReading],
) -> str:
    if not isinstance(readings, Sequence) or not readings:
        raise ValueError("complete reading universe is missing")
    by_frame: dict[str, list[lattice_planes.PlaneReading]] = {}
    seen_object_ids: set[int] = set()
    for reading in readings:
        if not isinstance(reading, lattice_planes.PlaneReading):
            raise ValueError("complete reading universe contains a foreign object")
        if id(reading) in seen_object_ids:
            raise ValueError("complete reading universe repeats an object")
        seen_object_ids.add(id(reading))
        if type(reading.frame) is not str or not reading.frame:
            raise ValueError("reading frame identity is invalid")
        by_frame.setdefault(reading.frame, []).append(reading)
    payload = []
    for frame in sorted(by_frame):
        try:
            ordered = tuple(sorted(by_frame[frame], key=lattice_planes.reading_order_key))
            keys = tuple(lattice_planes.reading_order_key(item) for item in ordered)
            if len(set(keys)) != len(keys):
                raise ValueError("complete reading universe has ambiguous order keys")
            for key in keys:
                _json_value(key)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("complete reading universe cannot be canonicalized") from error
        payload.extend({"frame": frame, "index": index, "reading_order_key": key}
                       for index, key in enumerate(keys))
    return _sha256_json(payload)


def _signature_for_endpoint(
    reading: lattice_planes.PlaneReading,
    raw_b: float,
    signatures: Sequence[lattice_planes.PlaneSignature],
) -> tuple[lattice_planes.PlaneSignature, float]:
    selected = reading.matching_alternatives(raw_b)
    if len(selected) != 1:
        raise ValueError("selected b alternative is missing or ambiguous")
    canonical_b = float(selected[0].blocks_per_period)
    matches = []
    foreign = False
    for signature in signatures:
        if not isinstance(signature, lattice_planes.PlaneSignature):
            raise ValueError("signature collection contains a foreign object")
        if signature.reading is not reading:
            candidate = signature.reading
            if (isinstance(candidate, lattice_planes.PlaneReading)
                    and candidate.frame == reading.frame
                    and lattice_planes.reading_order_key(candidate)
                    == lattice_planes.reading_order_key(reading)
                    and len(candidate.matching_alternatives(signature.blocks_per_period)) == 1):
                foreign = True
            continue
        own_alt = reading.matching_alternatives(signature.blocks_per_period)
        if len(own_alt) == 1 and own_alt[0].blocks_per_period == canonical_b:
            matches.append(signature)
    if foreign:
        raise ValueError("signature endpoint object differs from canonical reading")
    if len(matches) != 1:
        raise ValueError("exact endpoint signature is missing or ambiguous")
    signature = matches[0]
    if (signature.reading is not reading
            or float(signature.blocks_per_period) != canonical_b):
        raise ValueError("endpoint signature does not retain canonical b identity")
    return signature, canonical_b


def _match_record(match: plane_graph.PlaneMatch) -> dict[str, Any]:
    def number(value: Any) -> Any:
        if value is None:
            return None
        number_value = float(value)
        if math.isfinite(number_value):
            return number_value
        return {"float": ("+infinity" if number_value > 0 else
                          "-infinity" if number_value < 0 else "nan")}
    return {
        "left": match.left, "right": match.right,
        "left_axis": match.left_axis, "right_axis": match.right_axis,
        "left_b": number(match.left_b), "right_b": number(match.right_b),
        "quadrant_of_right": match.quadrant_of_right,
        "psr": ({"value": None, "positive_infinity": True}
                if match.psr == float("inf") else
                {"value": number(match.psr), "positive_infinity": False}),
        "peak": number(match.peak),
        "second_peak": number(match.second_peak),
        "overlap_cells": match.overlap_cells,
        "integer_residual": number(match.integer_residual),
        "score": number(match.score),
        "left_plane": match.left_plane, "right_plane": match.right_plane,
        "cell_shift_rc": (None if match.cell_shift_rc is None
                          else list(match.cell_shift_rc)),
    }


def _normal_for(
    support: plane_axis_support.C1AxisSupport,
    identity: plane_axis_support.C1MatchIdentity,
) -> plane_axis_support.NormalFaceConstraint:
    matches = [value for value in support.normal_face_constraints
               if value.match_identity == identity]
    if len(matches) != 1:
        raise ValueError("selected C1 normal-face constraint is missing or ambiguous")
    return matches[0]


def _mapped_axes_for(
    support: plane_axis_support.C1AxisSupport,
    identity: plane_axis_support.C1MatchIdentity,
) -> tuple[plane_axis_support.MappedCellShiftAxis,
           plane_axis_support.MappedCellShiftAxis]:
    values = [value for value in support.mapped_cell_shifts
              if value.match_identity == identity]
    if (len(values) != 2 or {value.right_local_axis for value in values} != {"u", "v"}
            or len({value.world_axis for value in values}) != 2):
        raise ValueError("C1 lateral axis mapping is missing, repeated or incompatible")
    return tuple(sorted(values, key=lambda value: value.world_axis))  # type: ignore[return-value]


def _axis_rows_for(
    identity: plane_axis_support.C1MatchIdentity,
    normal: plane_axis_support.NormalFaceConstraint,
    mapped_axes: Sequence[plane_axis_support.MappedCellShiftAxis],
    phase: plane_residue.HypothesisPhaseShiftEvidence,
) -> tuple[_AxisRow, _AxisRow]:
    phase_by = {(record.frame, record.local_axis): record
                for record in phase.phase_candidates}
    left_tangents = tuple(label for label in range(3) if label != identity.left_axis)
    right_tangents = tuple(label for label in range(3) if label != identity.right_axis)
    rows = []
    for mapped in mapped_axes:
        left_labels = [label for label in left_tangents
                       if plane_graph.axis_transform(
                           label, normal.left_frame_quadrant)[0] == mapped.world_axis]
        if len(left_labels) != 1:
            raise ValueError("a C1 world axis has no unique left tangent label")
        left_label = left_labels[0]
        left_axis, left_sign = plane_graph.axis_transform(
            left_label, normal.left_frame_quadrant)
        right_axis, right_sign = plane_graph.axis_transform(
            mapped.right_local_axis_label, normal.right_frame_quadrant)
        if ((left_axis, right_axis) != (mapped.world_axis, mapped.world_axis)
                or right_sign != mapped.right_world_sign
                or mapped.right_local_axis_label not in right_tangents):
            raise ValueError("C1 endpoint axes disagree with absolute quadrants")
        left_name = "u" if left_label == left_tangents[0] else "v"
        right_name = mapped.right_local_axis
        left_record = phase_by.get((identity.left_frame, left_name))
        right_record = phase_by.get((identity.right_frame, right_name))
        if (left_record is None or right_record is None
                or left_record.refusal is not None or right_record.refusal is not None
                or left_record.reading_axis != identity.left_axis
                or right_record.reading_axis != identity.right_axis
                or left_record.plane_index != identity.left_plane
                or right_record.plane_index != identity.right_plane):
            raise ValueError("P1-B record does not bind the mapped endpoint axis")
        rows.append(_AxisRow(
            mapped.world_axis, left_name, left_label, right_name,
            mapped.right_local_axis_label, left_sign, right_sign,
            left_record, right_record))
    rows.sort(key=lambda value: value.world_axis)
    if len(rows) != 2 or rows[0].world_axis == rows[1].world_axis \
            or normal.world_axis in {rows[0].world_axis, rows[1].world_axis}:
        raise ValueError("mapped world axes must be two distinct lateral axes")
    return rows[0], rows[1]


def _derive_physical_permutation(
    identity: plane_axis_support.C1MatchIdentity,
    normal: plane_axis_support.NormalFaceConstraint,
) -> np.ndarray:
    left_labels = tuple(label for label in range(3) if label != identity.left_axis)
    right_labels = tuple(label for label in range(3) if label != identity.right_axis)
    permutation = np.zeros((2, 2), dtype=np.float64)
    for right_index, right_label in enumerate(right_labels):
        world_axis, right_sign = plane_graph.axis_transform(
            right_label, normal.right_frame_quadrant)
        left_candidates = [label for label in left_labels
                           if plane_graph.axis_transform(
                               label, normal.left_frame_quadrant)[0] == world_axis]
        if len(left_candidates) != 1:
            raise ValueError("relative map does not bijectively preserve lateral world axes")
        left_label = left_candidates[0]
        left_index = left_labels.index(left_label)
        _axis, left_sign = plane_graph.axis_transform(
            left_label, normal.left_frame_quadrant)
        permutation[left_index, right_index] = float(left_sign * right_sign)
    if (np.count_nonzero(permutation, axis=0).tolist() != [1, 1]
            or np.count_nonzero(permutation, axis=1).tolist() != [1, 1]):
        raise ValueError("relative map is not a signed permutation")
    return permutation


def _build_binding(
    identity: plane_axis_support.C1MatchIdentity,
    c1_axes: Sequence[plane_axis_support.MappedCellShiftAxis],
    p1d_domain: plane_shift_domain.MatchShiftDomain,
    left_geometry: str,
    right_geometry: str,
    left_content: str,
    right_content: str,
    homography_sha: str,
) -> dict[str, Any]:
    return {
        "mapping_version": "block-cell-grid-v2",
        "replayed_homography_sha256": homography_sha,
        "c1_geometry_digests": (c1_axes[0].left_signature_geometry_sha256,
                                 c1_axes[0].right_signature_geometry_sha256),
        "recomputed_c1_geometry_digests": (left_geometry, right_geometry),
        "c1_homography_digests": tuple(axis.cell_homography_sha256
                                        for axis in c1_axes),
        "p1d_geometry_digests": (
            p1d_domain.left_signature_geometry_sha256,
            p1d_domain.right_signature_geometry_sha256),
        "recomputed_p1d_geometry_digests": (left_geometry, right_geometry),
        "p1d_content_digests": (
            p1d_domain.left_signature_content_sha256,
            p1d_domain.right_signature_content_sha256),
        "recomputed_p1d_content_digests": (left_content, right_content),
        "p1d_homography_digest": p1d_domain.cell_homography_sha256,
    }


def _prep_match(
    identity: plane_axis_support.C1MatchIdentity,
    match: plane_graph.PlaneMatch,
    support: plane_axis_support.C1AxisSupport,
    p1d_domain: plane_shift_domain.MatchShiftDomain,
    phase: plane_residue.HypothesisPhaseShiftEvidence,
    readings: Sequence[lattice_planes.PlaneReading],
    signatures: Sequence[lattice_planes.PlaneSignature],
    canonical_universe_sha: str,
    p1d_result: plane_shift_domain.ShiftDomainResult,
    c1_support_sha: str,
    tolerance: float,
) -> _PreparedMatch:
    normal = _normal_for(support, identity)
    mapped_axes = _mapped_axes_for(support, identity)
    if phase.refusal is not None or _phase_identity(phase) != identity:
        raise _BuildFailure("residue_candidate_invalid", "P1-B evidence identity/refusal is invalid")
    if phase.relative_shift != mapped_axes[0].support:
        raise _BuildFailure("provenance_mismatch", "P1-B shift support differs from C1 source")
    if mapped_axes[1].support != phase.relative_shift:
        raise _BuildFailure("provenance_mismatch", "C1 axes do not share exact P1-B support")
    try:
        ordered = tuple(sorted((value for value in readings
                                if value.frame == identity.left_frame),
                               key=lattice_planes.reading_order_key))
        left_reading = ordered[identity.left_plane]
        ordered_right = tuple(sorted((value for value in readings
                                      if value.frame == identity.right_frame),
                                     key=lattice_planes.reading_order_key))
        right_reading = ordered_right[identity.right_plane]
    except (IndexError, TypeError, ValueError, OverflowError) as error:
        raise _BuildFailure("provenance_mismatch", "canonical endpoint index cannot be resolved") from error
    if (left_reading.frame != identity.left_frame
            or right_reading.frame != identity.right_frame
            or left_reading.axis != identity.left_axis
            or right_reading.axis != identity.right_axis):
        raise _BuildFailure("provenance_mismatch", "canonical endpoint object/axis differs")
    if (match.left != identity.left_frame or match.right != identity.right_frame
            or match.left_plane != identity.left_plane
            or match.right_plane != identity.right_plane
            or match.left_axis != identity.left_axis
            or match.right_axis != identity.right_axis
            or match.left_b != identity.left_b or match.right_b != identity.right_b
            or match.quadrant_of_right != identity.quadrant_of_right):
        raise _BuildFailure("provenance_mismatch", "PlaneMatch content differs from C1 identity")
    try:
        left_signature, left_b = _signature_for_endpoint(
            left_reading, identity.left_b, signatures)
        right_signature, right_b = _signature_for_endpoint(
            right_reading, identity.right_b, signatures)
        left_geometry = plane_axis_support._signature_geometry_digest(
            left_signature, identity.left_plane)
        right_geometry = plane_axis_support._signature_geometry_digest(
            right_signature, identity.right_plane)
        left_content = plane_shift_domain._signature_content_sha256(
            left_signature, identity.left_plane, left_b)
        right_content = plane_shift_domain._signature_content_sha256(
            right_signature, identity.right_plane, right_b)
        homography = plane_graph.cell_grid_homography(
            left_signature, right_signature, identity.quadrant_of_right)
        homography_sha = plane_axis_support._grid_homography_digest(
            left_signature, right_signature, identity.quadrant_of_right,
            identity.left_plane, identity.right_plane, homography)
    except (AttributeError, TypeError, ValueError, IndexError, OverflowError,
            np.linalg.LinAlgError) as error:
        raise _BuildFailure("homography_incompatible", "current signature grid map cannot be replayed") from error
    if None in (left_geometry, right_geometry, left_content, right_content, homography_sha):
        raise _BuildFailure("provenance_mismatch", "current signature digest replay failed")
    left_geometry = str(left_geometry)
    right_geometry = str(right_geometry)
    left_content = str(left_content)
    right_content = str(right_content)
    homography_sha = str(homography_sha)
    if (left_geometry != p1d_domain.left_signature_geometry_sha256
            or right_geometry != p1d_domain.right_signature_geometry_sha256
            or left_content != p1d_domain.left_signature_content_sha256
            or right_content != p1d_domain.right_signature_content_sha256
            or homography_sha != p1d_domain.cell_homography_sha256
            or any((axis.left_signature_geometry_sha256 != left_geometry
                    or axis.right_signature_geometry_sha256 != right_geometry
                    or axis.cell_homography_sha256 != homography_sha)
                   for axis in mapped_axes)):
        raise _BuildFailure("provenance_mismatch", "recomputed C1/P1-D digests differ")
    permutation = _derive_physical_permutation(identity, normal)
    left_origin = tuple(float(value) for value in left_signature.origin_uv)
    right_origin = tuple(float(value) for value in right_signature.origin_uv)
    binding = _build_binding(identity, mapped_axes, p1d_domain,
                             left_geometry, right_geometry, left_content,
                             right_content, homography_sha)
    incompatibility = _finite_compatible_map(
        homography, permutation, left_origin, right_origin, tolerance, binding)
    if incompatibility is not None:
        raise _BuildFailure("homography_incompatible", incompatibility)
    axis_rows = _axis_rows_for(identity, normal, mapped_axes, phase)
    axis_pair_counts = tuple((row.world_axis,
                              len(row.left_record.residue_candidates)
                              * len(row.right_record.residue_candidates))
                             for row in axis_rows)
    compatibility_sha = _sha256_json(_compatibility_payload(
        homography, permutation, left_origin, right_origin, tolerance, binding))
    return _PreparedMatch(
        identity, normal, p1d_domain, phase, mapped_axes,
        left_geometry, right_geometry, homography_sha, left_content, right_content,
        left_signature, right_signature, left_b, right_b,
        tuple(float(value) for value in homography.reshape(-1)),
        tuple(float(value) for value in permutation.reshape(-1)),
        left_origin, right_origin, axis_rows, axis_pair_counts,
        _sha256_json(phase.to_record()), compatibility_sha)


class _BuildFailure(Exception):
    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail[:_MAX_DETAIL_LENGTH]


def _safe_invalid_tolerance(value: Any) -> str:
    if type(value) is bool:
        return "boolean"
    if type(value) in (int, float):
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError):
            return "numeric_invalid"
        if math.isnan(converted):
            return "nan"
        if math.isinf(converted):
            return "positive_infinity" if converted > 0 else "negative_infinity"
        if converted < 0.0:
            return "negative"
        return "numeric_invalid"
    return type(value).__name__


def _tolerance_is_valid(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value)) and value >= 0.0


def _compute_result_hashes(payload: dict[str, Any]) -> tuple[str, str]:
    input_digest = _sha256_json(payload["input"])
    result_digest = _sha256_json({key: value for key, value in payload.items()
                                  if key not in {"input_sha256", "result_sha256"}})
    return input_digest, result_digest


def _make_result(
    *, domains: Sequence[MatchLateralVectorDomain],
    refusals: Sequence[C2RVectorRefusal], budget: C2RVectorBudget,
    identities: tuple[plane_axis_support.C1MatchIdentity, ...],
    plans: tuple[_C2RMatchPlan, ...], planned: C2RVectorWork,
    represented: C2RVectorWork, canonical_universe_sha: str,
    c1_support_sha: str, p1d_input_sha: str, matches_sha: str,
    phase_digests: tuple[tuple[plane_axis_support.C1MatchIdentity, str], ...],
    tolerance: float | None, invalid_tolerance: str | None,
    status: str, budget_failure_reason: str | None,
) -> C2RVectorDomainResult:
    ordered_domains = tuple(sorted(domains, key=lambda value: _identity_key(value.match_identity)))
    ordered_refusals = tuple(sorted(refusals, key=_refusal_key))
    obj = object.__new__(C2RVectorDomainResult)
    fields = {
        "schema": C2R_VECTOR_DOMAIN_SCHEMA,
        "version": C2R_VECTOR_DOMAIN_VERSION,
        "correlation_algorithm_id": plane_graph.PLANE_CORRELATION_ALGORITHM,
        "domains": ordered_domains, "refusals": ordered_refusals,
        "budget": budget, "requested_match_identities": identities,
        "_preflight_plans": plans, "planned_work": planned,
        "represented_work": represented,
        "canonical_reading_universe_sha256": canonical_universe_sha,
        "c1_support_sha256": c1_support_sha,
        "p1d_input_sha256": p1d_input_sha,
        "selected_matches_sha256": matches_sha,
        "phase_evidence_sha256": phase_digests,
        "comparison_tolerance": tolerance,
        "invalid_tolerance_input": invalid_tolerance,
        "status": status, "budget_failure_reason": budget_failure_reason,
        "completeness_scope": _COMPLETENESS_SCOPE,
        "physical_truth_membership": _PHYSICAL_TRUTH_MEMBERSHIP,
    }
    for name, value in fields.items():
        object.__setattr__(obj, name, value)
    payload = _result_payload(obj)
    input_digest, result_digest = _compute_result_hashes(payload)
    object.__setattr__(obj, "input_sha256", input_digest)
    object.__setattr__(obj, "result_sha256", result_digest)
    C2RVectorDomainResult.__post_init__(obj)
    return obj


def _budget_exceeds(work: C2RVectorWork, budget: C2RVectorBudget) -> bool:
    limits = budget.to_record()
    return any(work.to_record()[name] > limits["max_" + name]
               for name in work.to_record())


def _make_axis_factor(
    prepared: _PreparedMatch, row: _AxisRow,
) -> LateralAxisFactor:
    mapped = [axis for axis in prepared.mapped_axes if axis.world_axis == row.world_axis]
    if len(mapped) != 1:
        raise ValueError("axis factor lacks one exact C1 mapped source")
    values = {
        "match_identity": prepared.identity,
        "normal_axis": prepared.normal.world_axis,
        "world_axis": row.world_axis,
        "left_world_sign": row.left_sign,
        "right_world_sign": row.right_sign,
        "left_frame_quadrant": prepared.normal.left_frame_quadrant,
        "right_frame_quadrant": prepared.normal.right_frame_quadrant,
        "left_local_axis": row.left_name,
        "left_local_axis_label": row.left_label,
        "right_local_axis": row.right_name,
        "right_local_axis_label": row.right_label,
        "right_shift_source_component": "column" if row.right_name == "u" else "row",
        "left_side": ResidueSideTable.from_record(
            row.left_record, endpoint_role="left", local_axis_label=row.left_label),
        "right_side": ResidueSideTable.from_record(
            row.right_record, endpoint_role="right", local_axis_label=row.right_label),
        "c1_axis_source_sha256": _mapped_axis_source_digest(mapped[0]),
        "residue_pair_count": (len(row.left_record.residue_candidates)
                                * len(row.right_record.residue_candidates)),
    }
    provisional = object.__new__(LateralAxisFactor)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "factor_sha256",
                       _sha256_json(_axis_factor_payload(provisional)))
    LateralAxisFactor.__post_init__(provisional)
    return provisional


def _make_match_domain(
    prepared: _PreparedMatch,
    c1_support_sha: str,
    p1d_input_sha: str,
    universe_sha: str,
    tolerance: float,
) -> MatchLateralVectorDomain:
    factors = tuple(_make_axis_factor(prepared, row) for row in prepared.axis_rows)
    axis_pairs = tuple((factor.world_axis, factor.residue_pair_count) for factor in factors)
    count = (len(prepared.p1d_domain.entries)
             * factors[0].residue_pair_count * factors[1].residue_pair_count)
    provisional = object.__new__(MatchLateralVectorDomain)
    values = {
        "match_identity": prepared.identity,
        "normal_constraint": prepared.normal,
        "p1d_domain": prepared.p1d_domain,
        "phase_evidence": prepared.phase,
        "c1_mapped_axes": prepared.mapped_axes,
        "c1_support_sha256": c1_support_sha,
        "p1d_input_sha256": p1d_input_sha,
        "p1b_evidence_sha256": prepared.phase_sha256,
        "canonical_reading_universe_sha256": universe_sha,
        "left_signature_geometry_sha256": prepared.left_signature_geometry_sha256,
        "right_signature_geometry_sha256": prepared.right_signature_geometry_sha256,
        "cell_homography_sha256": prepared.cell_homography_sha256,
        "left_signature_content_sha256": prepared.left_signature_content_sha256,
        "right_signature_content_sha256": prepared.right_signature_content_sha256,
        "comparison_tolerance": tolerance,
        "homography": prepared.homography,
        "physical_permutation": prepared.physical_permutation,
        "left_origin_uv": prepared.left_origin_uv,
        "right_origin_uv": prepared.right_origin_uv,
        "compatibility_sha256": prepared.compatibility_sha256,
        "axis_factors": factors,
        "joint_entry_count": len(prepared.p1d_domain.entries),
        "axis_residue_pair_counts": axis_pairs,
        "represented_candidate_count": count,
        "logical_provenance_reference_count": 7 * count + 1,
        "index_schema": _INDEX_SCHEMA,
        "reconstruction_version": _RECONSTRUCTION_VERSION,
        "ordering_identifier": _CANONICAL_ORDERING,
        "materialization_scope": "index_only",
        "completeness_scope": _COMPLETENESS_SCOPE,
        "physical_truth_membership": _PHYSICAL_TRUTH_MEMBERSHIP,
        "conditional_truth_predicates": (
            ("c1_provenance_replayed", "verified"),
            ("same_physical_plane_lattice", "unverified"),
            ("true_local_residues_in_p1b", "unverified"),
            ("true_integer_carries_in_p1d", "unverified"),
            ("p1f_grid_map_and_correlation_convention", "verified"),
        ),
    }
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "shared_provenance_sha256",
                       _sha256_json(_shared_provenance_payload(provisional)))
    object.__setattr__(provisional, "candidate_product_sha256",
                       _sha256_json(_candidate_product_payload(provisional)))
    object.__setattr__(provisional, "domain_sha256",
                       _sha256_json(_domain_payload(provisional)))
    MatchLateralVectorDomain.__post_init__(provisional)
    return provisional


def _current_match_maps(
    support: plane_axis_support.C1AxisSupport,
    matches: Sequence[plane_graph.PlaneMatch],
    phases: Sequence[plane_residue.HypothesisPhaseShiftEvidence],
    p1d_result: plane_shift_domain.ShiftDomainResult | None,
):
    mapped: dict[plane_axis_support.C1MatchIdentity,
                 list[plane_axis_support.MappedCellShiftAxis]] = {}
    for axis in support.mapped_cell_shifts:
        mapped.setdefault(axis.match_identity, []).append(axis)
    identities = tuple(sorted(mapped, key=_identity_key))
    matches_by: dict[plane_axis_support.C1MatchIdentity, list[plane_graph.PlaneMatch]] = {}
    bad_match_rows = False
    for match in matches if isinstance(matches, Sequence) else ():
        if not isinstance(match, plane_graph.PlaneMatch):
            bad_match_rows = True
            continue
        try:
            identity = plane_axis_support.C1MatchIdentity.from_match(match)
        except (TypeError, ValueError, OverflowError):
            bad_match_rows = True
            continue
        if identity in mapped:
            matches_by.setdefault(identity, []).append(match)
    if not isinstance(matches, Sequence):
        bad_match_rows = True
    phases_by: dict[plane_axis_support.C1MatchIdentity,
                    list[plane_residue.HypothesisPhaseShiftEvidence]] = {}
    bad_phase_rows = False
    for phase in phases if isinstance(phases, Sequence) else ():
        if not isinstance(phase, plane_residue.HypothesisPhaseShiftEvidence):
            bad_phase_rows = True
            continue
        relative = phase.relative_shift
        if (not isinstance(relative, plane_residue.RelativeShiftSupport)
                or relative.schema != plane_residue.RELATIVE_SHIFT_SUPPORT_SCHEMA
                or relative.version != plane_residue.RELATIVE_SHIFT_SUPPORT_VERSION
                or relative.correlation_algorithm_id
                != plane_graph.PLANE_CORRELATION_ALGORITHM):
            bad_phase_rows = True
            continue
        try:
            identity = _phase_identity(phase)
        except (TypeError, ValueError, OverflowError):
            bad_phase_rows = True
            continue
        if identity in mapped:
            phases_by.setdefault(identity, []).append(phase)
    if not isinstance(phases, Sequence):
        bad_phase_rows = True
    domains_by = ({domain.match_identity: domain for domain in p1d_result.domains}
                  if isinstance(p1d_result, plane_shift_domain.ShiftDomainResult)
                  else {})
    return identities, mapped, matches_by, phases_by, domains_by, bad_match_rows, bad_phase_rows


def build_lateral_vector_domains(
    support: plane_axis_support.C1AxisSupport,
    p1d_result: plane_shift_domain.ShiftDomainResult,
    matches: Sequence[plane_graph.PlaneMatch],
    phase_evidence: Sequence[plane_residue.HypothesisPhaseShiftEvidence],
    *,
    readings: Sequence[lattice_planes.PlaneReading],
    signatures: Sequence[lattice_planes.PlaneSignature],
    comparison_tolerance: float,
    budget: C2RVectorBudget,
) -> C2RVectorDomainResult:
    """Index every compatible P1-D/P1-B provenance tuple atomically."""
    if not isinstance(budget, C2RVectorBudget):
        raise ValueError("an explicit valid C2RVectorBudget is required")
    if not isinstance(support, plane_axis_support.C1AxisSupport) or support.refusal is not None:
        empty = C2RVectorWork(0, 0, 0, 0, 0)
        placeholder = "0" * 64
        refusal = C2RVectorRefusal(None, "c1_status_invalid",
                                   "accepted C1 support is required")
        return _make_result(
            domains=(), refusals=(refusal,), budget=budget, identities=(), plans=(),
            planned=empty, represented=empty, canonical_universe_sha=placeholder,
            c1_support_sha=placeholder, p1d_input_sha=placeholder,
            matches_sha=placeholder, phase_digests=(), tolerance=None,
            invalid_tolerance="not_examined", status="refused",
            budget_failure_reason=None)
    mapped_by: dict[plane_axis_support.C1MatchIdentity,
                    list[plane_axis_support.MappedCellShiftAxis]] = {}
    for axis in support.mapped_cell_shifts:
        mapped_by.setdefault(axis.match_identity, []).append(axis)
    identities = tuple(sorted(mapped_by, key=_identity_key))
    if not identities:
        empty = C2RVectorWork(0, 0, 0, 0, 0)
        digest = _sha256_json(support.to_record())
        refusal = C2RVectorRefusal(None, "requested_identity_missing",
                                   "C1 support contains no mapped lateral match identity")
        return _make_result(
            domains=(), refusals=(refusal,), budget=budget, identities=(), plans=(),
            planned=empty, represented=empty, canonical_universe_sha="0" * 64,
            c1_support_sha=digest, p1d_input_sha="0" * 64,
            matches_sha="0" * 64, phase_digests=(), tolerance=None,
            invalid_tolerance="no_requested_identity", status="refused",
            budget_failure_reason=None)

    c1_support_sha = _sha256_json(support.to_record())
    p1d_input_sha = (p1d_result.input_sha256
                     if isinstance(p1d_result, plane_shift_domain.ShiftDomainResult)
                     else "0" * 64)
    try:
        universe_sha = _canonical_reading_universe_sha256(readings)
    except (TypeError, ValueError, OverflowError):
        universe_sha = _sha256_json({"invalid_reading_universe": True})
    tolerance_valid = _tolerance_is_valid(comparison_tolerance)
    tolerance = float(comparison_tolerance) if tolerance_valid else None
    invalid_tolerance = None if tolerance_valid else _safe_invalid_tolerance(comparison_tolerance)

    identities2, mapped_by, matches_by, phases_by, domains_by, bad_matches, bad_phases = \
        _current_match_maps(
            support, matches, phase_evidence,
            p1d_result if isinstance(p1d_result, plane_shift_domain.ShiftDomainResult)
            else None)
    del identities2
    selected_match_records = []
    phase_digests: list[tuple[plane_axis_support.C1MatchIdentity, str]] = []
    for identity in identities:
        rows = matches_by.get(identity, [])
        if len(rows) == 1:
            selected_match_records.append({"identity": _identity_record(identity),
                                           "match": _match_record(rows[0])})
        else:
            selected_match_records.append({"identity": _identity_record(identity),
                                           "match_count": len(rows)})
        phase_rows = phases_by.get(identity, [])
        phase_digest = (_sha256_json(phase_rows[0].to_record()) if len(phase_rows) == 1
                        else _sha256_json({"phase_count": len(phase_rows)}))
        phase_digests.append((identity, phase_digest))
    matches_sha = _sha256_json(selected_match_records)
    phase_digests_tuple = tuple(phase_digests)

    p1d_valid = (isinstance(p1d_result, plane_shift_domain.ShiftDomainResult)
                 and p1d_result.schema == plane_shift_domain.SHIFT_DOMAIN_SCHEMA
                 and p1d_result.version == plane_shift_domain.SHIFT_DOMAIN_VERSION
                 and p1d_result.correlation_algorithm_id
                 == plane_graph.PLANE_CORRELATION_ALGORITHM
                 and p1d_result.status == "complete"
                 and not p1d_result.refusals
                 and tuple(domain.match_identity for domain in p1d_result.domains)
                 == identities
                 and p1d_result.requested_match_identities == identities)
    if p1d_valid:
        radii = {domain.max_shift_cells for domain in p1d_result.domains}
        p1d_valid = (len(radii) == 1 and not bad_matches and not bad_phases
                     and all(
                         domain.schema == plane_shift_domain.SHIFT_DOMAIN_SCHEMA
                         and domain.version == plane_shift_domain.SHIFT_DOMAIN_VERSION
                         and domain.correlation_algorithm_id
                         == plane_graph.PLANE_CORRELATION_ALGORITHM
                         for domain in p1d_result.domains))
        if p1d_valid:
            try:
                p1d_valid = plane_shift_domain.validate_shift_domain_inputs(
                    p1d_result, support, tuple(match for rows in matches_by.values()
                                               for match in rows), signatures,
                    readings=readings, max_shift_cells=next(iter(radii)))
            except (TypeError, ValueError, AttributeError, OverflowError):
                p1d_valid = False

    prepared_by: dict[plane_axis_support.C1MatchIdentity, _PreparedMatch] = {}
    plans: list[_C2RMatchPlan] = []
    refusals: list[C2RVectorRefusal] = []
    for identity in identities:
        reason: str | None = None
        detail: str | None = None
        domain = domains_by.get(identity)
        entry_count = len(domain.entries) if domain is not None else 0
        pair_counts: tuple[tuple[int, int], ...] = ()
        phase_rows = phases_by.get(identity, [])
        matches_for_id = matches_by.get(identity, [])
        if not tolerance_valid:
            reason, detail = "formula_replay_mismatch", "comparison tolerance must be finite and non-negative"
        elif bad_matches:
            reason, detail = "requested_identity_missing", "match collection contains an invalid row"
        elif len(matches_for_id) == 0:
            reason, detail = "requested_identity_missing", "selected C1 match was not supplied"
        elif len(matches_for_id) != 1:
            reason, detail = "requested_identity_ambiguous", "selected C1 match is duplicated"
        elif not p1d_valid:
            reason, detail = "p1d_status_invalid", "complete P1-D result did not replay for current inputs"
        elif domain is None:
            reason, detail = "requested_identity_missing", "matching P1-D domain is absent"
        elif domain.match_identity != identity or not domain.entries:
            reason, detail = "joint_shift_invalid", "P1-D joint entry set is empty or mismatched"
        elif bad_phases:
            reason, detail = "residue_candidate_invalid", "phase evidence collection contains an invalid row"
        elif len(phase_rows) == 0:
            reason, detail = "residue_evidence_missing", "exact selected P1-B evidence is absent"
        elif len(phase_rows) != 1:
            reason, detail = "residue_evidence_ambiguous", "selected P1-B evidence is duplicated"
        elif phase_rows[0].relative_shift.psr != matches_for_id[0].psr:
            reason, detail = "provenance_mismatch", "P1-B PSR differs from selected match"
        else:
            try:
                prepared = _prep_match(
                    identity, matches_for_id[0], support, domain, phase_rows[0],
                    readings, signatures, universe_sha, p1d_result, c1_support_sha,
                    float(tolerance))
                prepared_by[identity] = prepared
                pair_counts = prepared.axis_pair_counts
            except _BuildFailure as error:
                reason, detail = error.reason, error.detail
            except (AttributeError, TypeError, ValueError, IndexError,
                    OverflowError, FloatingPointError) as error:
                reason, detail = "axis_mapping_invalid", str(error)[:_MAX_DETAIL_LENGTH]
        plans.append(_C2RMatchPlan(
            identity, entry_count, pair_counts, reason, detail))
        if reason is not None:
            refusals.append(C2RVectorRefusal(identity, reason, detail or reason))
    plans_tuple = tuple(plans)
    planned = _work_from_plans(plans_tuple)
    if _budget_exceeds(planned, budget):
        zero = C2RVectorWork(0, 0, 0, 0, 0)
        return _make_result(
            domains=(), refusals=(), budget=budget, identities=identities,
            plans=plans_tuple, planned=planned, represented=zero,
            canonical_universe_sha=universe_sha, c1_support_sha=c1_support_sha,
            p1d_input_sha=p1d_input_sha, matches_sha=matches_sha,
            phase_digests=phase_digests_tuple, tolerance=tolerance,
            invalid_tolerance=invalid_tolerance, status="budget_exhausted",
            budget_failure_reason="planned_budget_exceeded")
    if refusals:
        zero = C2RVectorWork(0, 0, 0, 0, 0)
        return _make_result(
            domains=(), refusals=refusals, budget=budget, identities=identities,
            plans=plans_tuple, planned=planned, represented=zero,
            canonical_universe_sha=universe_sha, c1_support_sha=c1_support_sha,
            p1d_input_sha=p1d_input_sha, matches_sha=matches_sha,
            phase_digests=phase_digests_tuple, tolerance=tolerance,
            invalid_tolerance=invalid_tolerance, status="refused",
            budget_failure_reason=None)

    # Store only the source factors and exact mixed-radix product dimensions.
    # Construction, serialization and replay never create candidate leaves.
    try:
        domains = tuple(_make_match_domain(
            prepared_by[identity], c1_support_sha, p1d_input_sha,
            universe_sha, float(tolerance)) for identity in identities)
    except (AttributeError, TypeError, ValueError, IndexError, OverflowError,
            FloatingPointError, np.linalg.LinAlgError):
        domains = ()
    represented = _work_from_domains(domains) if domains else C2RVectorWork(0, 0, 0, 0, 0)
    if represented != planned:
        zero = C2RVectorWork(0, 0, 0, 0, 0)
        return _make_result(
            domains=(), refusals=(), budget=budget, identities=identities,
            plans=plans_tuple, planned=planned, represented=zero,
            canonical_universe_sha=universe_sha, c1_support_sha=c1_support_sha,
            p1d_input_sha=p1d_input_sha, matches_sha=matches_sha,
            phase_digests=phase_digests_tuple, tolerance=tolerance,
            invalid_tolerance=invalid_tolerance, status="budget_exhausted",
            budget_failure_reason="runtime_count_mismatch")
    return _make_result(
        domains=domains, refusals=(), budget=budget, identities=identities,
        plans=plans_tuple, planned=planned, represented=represented,
        canonical_universe_sha=universe_sha, c1_support_sha=c1_support_sha,
        p1d_input_sha=p1d_input_sha, matches_sha=matches_sha,
        phase_digests=phase_digests_tuple, tolerance=tolerance,
        invalid_tolerance=invalid_tolerance, status="complete",
        budget_failure_reason=None)


def validate_lateral_vector_domain_inputs(
    result: C2RVectorDomainResult,
    support: plane_axis_support.C1AxisSupport,
    p1d_result: plane_shift_domain.ShiftDomainResult,
    matches: Sequence[plane_graph.PlaneMatch],
    phase_evidence: Sequence[plane_residue.HypothesisPhaseShiftEvidence],
    *,
    readings: Sequence[lattice_planes.PlaneReading],
    signatures: Sequence[lattice_planes.PlaneSignature],
    comparison_tolerance: float,
) -> bool:
    """Recompute canonical source digests and the complete candidate result."""
    if (not isinstance(result, C2RVectorDomainResult)
            or not _tolerance_is_valid(comparison_tolerance)
            or result.comparison_tolerance != float(comparison_tolerance)):
        return False
    try:
        current = build_lateral_vector_domains(
            support, p1d_result, matches, phase_evidence, readings=readings,
            signatures=signatures, comparison_tolerance=comparison_tolerance,
            budget=result.budget)
        return current.to_json_bytes() == result.to_json_bytes()
    except (AttributeError, TypeError, ValueError, OverflowError, FloatingPointError):
        return False
