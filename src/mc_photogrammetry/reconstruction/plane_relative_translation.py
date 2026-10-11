"""Exact, bounded graph-wide relative-translation solving for C2-R-B.

This module consumes complete indexed vector-domain inputs.  It records local
component gauges and cycle evidence; it does not place a world model.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from fractions import Fraction
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from mc_photogrammetry.reconstruction import plane_axis_support
from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction import plane_residue
from mc_photogrammetry.reconstruction import plane_shift_domain
from mc_photogrammetry.reconstruction import plane_vector_domain


_STATUS = frozenset({"complete", "refused", "budget_exhausted"})
_OUTCOMES = frozenset({
    "unique_validated", "unique_unvalidated_tree", "ambiguous", "contradiction",
})
_READINESS = frozenset({"ready", "review_required", "unavailable"})
_REFUSALS = frozenset({
    "vector_result_invalid", "vector_result_replay_mismatch",
    "comparison_tolerance_mismatch", "closure_tolerance_invalid",
    "duplicate_match_identity", "identity_endpoint_invalid", "nonfinite_delta",
    "factor_contract_invalid", "graph_empty",
})


def _digest_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_digest(value: Any) -> bool:
    return (type(value) is str and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _finite_float(value: Any) -> float | None:
    if type(value) not in (int, float):
        return None
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _dyadic_from_float(value: float) -> tuple[int, int]:
    """Return the canonical exact rational represented by one finite float."""
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("dyadic source must be a finite float")
    numerator, denominator = value.as_integer_ratio()
    return numerator, denominator


def _validate_dyadic(value: Any) -> bool:
    if (type(value) is not tuple or len(value) != 2
            or type(value[0]) is not int or type(value[1]) is not int
            or value[1] <= 0 or value[1] & (value[1] - 1) != 0):
        return False
    return math.gcd(abs(value[0]), value[1]) == 1


def _fraction(value: tuple[int, int]) -> Fraction:
    if not _validate_dyadic(value):
        raise ValueError("dyadic must be reduced with a power-of-two denominator")
    return Fraction(value[0], value[1])


def _dyadic(value: Fraction) -> tuple[int, int]:
    numerator, denominator = value.numerator, value.denominator
    if denominator & (denominator - 1):
        raise ValueError("exact translation left the dyadic rationals")
    return numerator, denominator


def _identity_key(identity: plane_axis_support.C1MatchIdentity) -> tuple[Any, ...]:
    return identity.sort_key()


@dataclass(frozen=True, slots=True)
class C2RBudget:
    """Explicit positive whole-request deterministic and runtime caps."""

    max_components: int
    max_frames: int
    max_edges: int
    max_tree_edges: int
    max_chords: int
    max_cycle_rows: int
    max_p1d_entries: int
    max_residue_values: int
    max_candidate_views_scanned: int
    max_partial_assignments: int
    max_closure_component_checks: int
    max_membership_probes: int
    max_memoized_terms: int
    max_provenance_witnesses: int
    max_certificates: int

    def __post_init__(self) -> None:
        names = tuple(self.to_record())
        for name in names:
            value = getattr(self, name)
            minimum = 2 if name in {
                "max_provenance_witnesses", "max_certificates"} else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an explicit positive integer cap")

    def to_record(self) -> dict[str, int]:
        return {
            "max_components": self.max_components,
            "max_frames": self.max_frames,
            "max_edges": self.max_edges,
            "max_tree_edges": self.max_tree_edges,
            "max_chords": self.max_chords,
            "max_cycle_rows": self.max_cycle_rows,
            "max_p1d_entries": self.max_p1d_entries,
            "max_residue_values": self.max_residue_values,
            "max_candidate_views_scanned": self.max_candidate_views_scanned,
            "max_partial_assignments": self.max_partial_assignments,
            "max_closure_component_checks": self.max_closure_component_checks,
            "max_membership_probes": self.max_membership_probes,
            "max_memoized_terms": self.max_memoized_terms,
            "max_provenance_witnesses": self.max_provenance_witnesses,
            "max_certificates": self.max_certificates,
        }


@dataclass(frozen=True, slots=True)
class C2RBDeterministicWork:
    components: int
    frames: int
    edges: int
    tree_edges: int
    chords: int
    cycle_rows: int
    p1d_entries: int
    residue_values: int
    represented_candidate_count: int

    def __post_init__(self) -> None:
        if any(type(getattr(self, field)) is not int or getattr(self, field) < 0
               for field in self.__dataclass_fields__):
            raise ValueError("deterministic work counters must be non-negative integers")
        if (self.tree_edges + self.chords != self.edges
                or self.cycle_rows != 3 * self.chords
                or self.components > self.frames):
            raise ValueError("deterministic graph work counts are inconsistent")

    def to_record(self) -> dict[str, int]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class C2RBRuntimeWork:
    candidate_views_scanned: int
    partial_assignments: int
    closure_component_checks: int
    membership_probes: int
    memoized_terms: int
    provenance_witnesses: int
    candidate_at_calls: int
    certificates_retained: int

    def __post_init__(self) -> None:
        if any(type(getattr(self, field)) is not int or getattr(self, field) < 0
               for field in self.__dataclass_fields__):
            raise ValueError("runtime work counters must be non-negative integers")
        if self.candidate_at_calls != 0:
            raise ValueError("the solver must never materialize A4 candidates")

    def to_record(self) -> dict[str, int]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class C2RBRefusal:
    reason: str
    detail: str
    match_identity: plane_axis_support.C1MatchIdentity | None = None

    def __post_init__(self) -> None:
        if (self.reason not in _REFUSALS or type(self.detail) is not str
                or not self.detail or len(self.detail) > 240
                or any(ord(char) < 32 or ord(char) == 127 for char in self.detail)
                or (self.match_identity is not None
                    and not isinstance(self.match_identity,
                                       plane_axis_support.C1MatchIdentity))):
            raise ValueError("C2-R-B refusal is malformed")

    def to_record(self) -> dict[str, Any]:
        return {"reason": self.reason, "detail": self.detail,
                "match_identity": (None if self.match_identity is None
                                   else self.match_identity.to_record())}


def _ref_identity_key(value: C2RBRefusal) -> tuple[Any, ...]:
    identity = value.match_identity
    return (() if identity is None else _identity_key(identity), value.reason, value.detail)


def _identity_record(value: plane_axis_support.C1MatchIdentity) -> dict[str, Any]:
    return value.to_record()


def _dyadics_record(values: Sequence[tuple[int, int]]) -> list[list[int]]:
    return [[numerator, denominator] for numerator, denominator in values]


def _float_tuple_record(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values]


def _edge_ref_payload(value: "C2RBEdgeCandidateRef") -> dict[str, Any]:
    return {
        "match_identity": _identity_record(value.match_identity),
        "domain_sha256": value.domain_sha256,
        "global_index": value.global_index,
        "p1d_entry_index": value.p1d_entry_index,
        "shift_entry_rc": list(value.shift_entry_rc),
        "axis_worlds": list(value.axis_worlds),
        "residue_side_sha256s": list(value.residue_side_sha256s),
        "residue_candidate_counts": list(value.residue_candidate_counts),
        "p1d_entry_count": value.p1d_entry_count,
        "p1d_max_shift_cells": value.p1d_max_shift_cells,
        "residue_indices": list(value.residue_indices),
        "provenance_key": value.provenance_key,
        "delta_dyadic_xyz": _dyadics_record(value.delta_dyadic_xyz),
        "delta_xyz": _float_tuple_record(value.delta_xyz),
        "tree_delta_class_multiplicity": value.tree_delta_class_multiplicity,
        "chord_satisfying_provenance_count": value.chord_satisfying_provenance_count,
    }


@dataclass(frozen=True, slots=True)
class C2RBEdgeCandidateRef:
    match_identity: plane_axis_support.C1MatchIdentity
    domain_sha256: str
    global_index: int
    p1d_entry_index: int
    shift_entry_rc: tuple[int, int]
    axis_worlds: tuple[int, int]
    residue_side_sha256s: tuple[str, str, str, str]
    residue_candidate_counts: tuple[int, int, int, int]
    p1d_entry_count: int
    p1d_max_shift_cells: int
    residue_indices: tuple[int, int, int, int]
    provenance_key: tuple[Any, ...]
    delta_dyadic_xyz: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    delta_xyz: tuple[float, float, float]
    tree_delta_class_multiplicity: int | None
    chord_satisfying_provenance_count: int | None
    ref_sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(self.match_identity, plane_axis_support.C1MatchIdentity)
                or not _is_digest(self.domain_sha256)
                or type(self.global_index) is not int or self.global_index < 0
                or type(self.p1d_entry_index) is not int or self.p1d_entry_index < 0
                or type(self.shift_entry_rc) is not tuple or len(self.shift_entry_rc) != 2
                or any(type(value) is not int for value in self.shift_entry_rc)
                or type(self.axis_worlds) is not tuple or len(self.axis_worlds) != 2
                or any(type(value) is not int or value not in (0, 1, 2)
                       for value in self.axis_worlds)
                or self.axis_worlds[0] == self.axis_worlds[1]
                or type(self.residue_side_sha256s) is not tuple
                or len(self.residue_side_sha256s) != 4
                or any(not _is_digest(value) for value in self.residue_side_sha256s)
                or type(self.residue_candidate_counts) is not tuple
                or len(self.residue_candidate_counts) != 4
                or any(type(value) is not int or value < 1
                       for value in self.residue_candidate_counts)
                or type(self.p1d_entry_count) is not int or self.p1d_entry_count < 1
                or type(self.p1d_max_shift_cells) is not int
                or self.p1d_max_shift_cells < 1
                or any(abs(value) > self.p1d_max_shift_cells
                       for value in self.shift_entry_rc)
                or type(self.residue_indices) is not tuple or len(self.residue_indices) != 4
                or any(type(value) is not int or value < 0 for value in self.residue_indices)
                or type(self.provenance_key) is not tuple
                or type(self.delta_dyadic_xyz) is not tuple
                or len(self.delta_dyadic_xyz) != 3
                or any(not _validate_dyadic(value) for value in self.delta_dyadic_xyz)
                or type(self.delta_xyz) is not tuple or len(self.delta_xyz) != 3
                or any(type(value) is not float or not math.isfinite(value)
                       for value in self.delta_xyz)):
            raise ValueError("C2-R-B edge candidate reference is malformed")
        if (type(self.provenance_key) is not tuple or len(self.provenance_key) != 2
                or type(self.provenance_key[0]) is not tuple
                or len(self.provenance_key[0]) != 2
                or any(type(value) is not int for value in self.provenance_key[0])
                or self.provenance_key[0] != self.shift_entry_rc
                or type(self.provenance_key[1]) is not tuple
                or len(self.provenance_key[1]) != 2):
            raise ValueError("edge provenance does not have the exact A4 key shape")
        flattened_indices: list[int] = []
        for factor_index, item in enumerate(self.provenance_key[1]):
            if (type(item) is not tuple or len(item) != 5
                    or type(item[0]) is not int
                    or item[0] != self.axis_worlds[factor_index]
                    or item[1] != self.residue_side_sha256s[2 * factor_index]
                    or item[3] != self.residue_side_sha256s[2 * factor_index + 1]
                    or type(item[2]) is not int or type(item[4]) is not int
                    or item[2] < 0
                    or item[2] >= self.residue_candidate_counts[2 * factor_index]
                    or item[4] < 0
                    or item[4] >= self.residue_candidate_counts[2 * factor_index + 1]):
                raise ValueError("edge provenance does not replay its A4 residue factors")
            flattened_indices.extend((item[2], item[4]))
        per_entry = math.prod(self.residue_candidate_counts)
        if (tuple(flattened_indices) != self.residue_indices
                or self.global_index >= self.p1d_entry_count * per_entry
                or self.global_index // per_entry != self.p1d_entry_index
                or self.p1d_entry_index >= self.p1d_entry_count):
            raise ValueError("edge provenance indices do not resolve its A4 product position")
        if tuple(_dyadic_from_float(value) for value in self.delta_xyz) != self.delta_dyadic_xyz:
            raise ValueError("edge reference dyadics do not replay the rounded A4 floats")
        tree_count = self.tree_delta_class_multiplicity
        chord_count = self.chord_satisfying_provenance_count
        if ((tree_count is None) == (chord_count is None)
                or (tree_count is not None
                    and (type(tree_count) is not int or tree_count < 1))
                or (chord_count is not None
                    and (type(chord_count) is not int or chord_count < 1))):
            raise ValueError("edge reference requires exactly one positive multiplicity")
        if (not _is_digest(self.ref_sha256)
                or _digest_json(_edge_ref_payload(self)) != self.ref_sha256):
            raise ValueError("edge reference digest does not bind its contents")

    def to_record(self) -> dict[str, Any]:
        record = _edge_ref_payload(self)
        record["ref_sha256"] = self.ref_sha256
        return record


def _cycle_payload(value: "C2RBCycleResidual") -> dict[str, Any]:
    return {
        "chord_identity": _identity_record(value.chord_identity),
        "required_chord_delta_dyadic_xyz": _dyadics_record(
            value.required_chord_delta_dyadic_xyz),
        "residual_bounds_dyadic_xyz": [
            list(pair) for pair in value.residual_bounds_dyadic_xyz],
        "matching_provenance_count": value.matching_provenance_count,
    }


@dataclass(frozen=True, slots=True)
class C2RBCycleResidual:
    chord_identity: plane_axis_support.C1MatchIdentity
    required_chord_delta_dyadic_xyz: tuple[
        tuple[int, int], tuple[int, int], tuple[int, int]]
    residual_bounds_dyadic_xyz: tuple[
        tuple[tuple[int, int], tuple[int, int]],
        tuple[tuple[int, int], tuple[int, int]],
        tuple[tuple[int, int], tuple[int, int]],
    ]
    matching_provenance_count: int
    cycle_sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(self.chord_identity, plane_axis_support.C1MatchIdentity)
                or type(self.required_chord_delta_dyadic_xyz) is not tuple
                or len(self.required_chord_delta_dyadic_xyz) != 3
                or any(not _validate_dyadic(value)
                       for value in self.required_chord_delta_dyadic_xyz)
                or type(self.residual_bounds_dyadic_xyz) is not tuple
                or len(self.residual_bounds_dyadic_xyz) != 3
                or any(type(pair) is not tuple or len(pair) != 2
                       or any(not _validate_dyadic(bound) for bound in pair)
                       or _fraction(pair[0]) > _fraction(pair[1])
                       for pair in self.residual_bounds_dyadic_xyz)
                or type(self.matching_provenance_count) is not int
                or self.matching_provenance_count < 0):
            raise ValueError("C2-R-B cycle residual is malformed")
        if (not _is_digest(self.cycle_sha256)
                or _digest_json(_cycle_payload(self)) != self.cycle_sha256):
            raise ValueError("cycle digest does not bind its contents")

    def to_record(self) -> dict[str, Any]:
        record = _cycle_payload(self)
        record["cycle_sha256"] = self.cycle_sha256
        return record


def _certificate_payload(value: "C2RBTranslationClassCertificate") -> dict[str, Any]:
    return {
        "gauge_anchor": value.gauge_anchor,
        "translations_dyadic_xyz": [
            {"frame": frame, "xyz": _dyadics_record(xyz)}
            for frame, xyz in value.translations_dyadic_xyz],
        "edge_references": [reference.to_record() for reference in value.edge_references],
        "assignment_provenance_multiplicity": value.assignment_provenance_multiplicity,
    }


@dataclass(frozen=True, slots=True)
class C2RBTranslationClassCertificate:
    gauge_anchor: str
    translations_dyadic_xyz: tuple[
        tuple[str, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]], ...]
    edge_references: tuple[C2RBEdgeCandidateRef, ...]
    assignment_provenance_multiplicity: int
    certificate_sha256: str

    def __post_init__(self) -> None:
        if (type(self.gauge_anchor) is not str or not self.gauge_anchor
                or type(self.translations_dyadic_xyz) is not tuple
                or not self.translations_dyadic_xyz
                or tuple(frame for frame, _xyz in self.translations_dyadic_xyz)
                != tuple(sorted(frame for frame, _xyz in self.translations_dyadic_xyz))
                or self.translations_dyadic_xyz[0][0] != self.gauge_anchor
                or any(type(xyz) is not tuple or len(xyz) != 3
                       or any(not _validate_dyadic(value) for value in xyz)
                       for _frame, xyz in self.translations_dyadic_xyz)
                or self.translations_dyadic_xyz[0][1] != ((0, 1), (0, 1), (0, 1))
                or type(self.edge_references) is not tuple
                or any(not isinstance(ref, C2RBEdgeCandidateRef)
                       for ref in self.edge_references)
                or len({ref.match_identity for ref in self.edge_references})
                   != len(self.edge_references)
                or type(self.assignment_provenance_multiplicity) is not int
                or self.assignment_provenance_multiplicity < 1):
            raise ValueError("translation class certificate is malformed")
        if (not _is_digest(self.certificate_sha256)
                or _digest_json(_certificate_payload(self)) != self.certificate_sha256):
            raise ValueError("translation certificate digest does not bind its contents")

    @property
    def translation_key(self) -> tuple[Any, ...]:
        return self.translations_dyadic_xyz

    def to_record(self) -> dict[str, Any]:
        record = _certificate_payload(self)
        record["certificate_sha256"] = self.certificate_sha256
        return record


def _component_payload(value: "C2RBComponentResult") -> dict[str, Any]:
    return {
        "frames": list(value.frames), "gauge_anchor": value.gauge_anchor,
        "tree_edges": [_identity_record(identity) for identity in value.tree_edges],
        "chords": [_identity_record(identity) for identity in value.chords],
        "cycle_rank": value.cycle_rank,
        "outcome": value.outcome,
        "translation_classes": [certificate.to_record()
                                for certificate in value.translation_classes],
        "cycle_residuals": [cycle.to_record() for cycle in value.cycle_residuals],
        "failing_chords": [_identity_record(identity) for identity in value.failing_chords],
        "ambiguity_class_count_lower_bound": value.ambiguity_class_count_lower_bound,
        "exhaustive": value.exhaustive,
    }


@dataclass(frozen=True, slots=True)
class C2RBComponentResult:
    frames: tuple[str, ...]
    gauge_anchor: str
    tree_edges: tuple[plane_axis_support.C1MatchIdentity, ...]
    chords: tuple[plane_axis_support.C1MatchIdentity, ...]
    cycle_rank: int
    outcome: str
    translation_classes: tuple[C2RBTranslationClassCertificate, ...]
    cycle_residuals: tuple[C2RBCycleResidual, ...]
    failing_chords: tuple[plane_axis_support.C1MatchIdentity, ...]
    ambiguity_class_count_lower_bound: int
    exhaustive: bool
    component_sha256: str

    def __post_init__(self) -> None:
        if (type(self.frames) is not tuple or not self.frames
                or any(type(frame) is not str or not frame for frame in self.frames)
                or tuple(sorted(set(self.frames))) != self.frames
                or self.gauge_anchor != self.frames[0]
                or type(self.tree_edges) is not tuple
                or type(self.chords) is not tuple
                or any(not isinstance(value, plane_axis_support.C1MatchIdentity)
                       for value in (*self.tree_edges, *self.chords))
                or len(set((*self.tree_edges, *self.chords)))
                   != len(self.tree_edges) + len(self.chords)
                or type(self.cycle_rank) is not int or self.cycle_rank != len(self.chords)
                or self.cycle_rank < 0 or self.outcome not in _OUTCOMES
                or type(self.translation_classes) is not tuple
                or any(not isinstance(value, C2RBTranslationClassCertificate)
                       for value in self.translation_classes)
                or type(self.cycle_residuals) is not tuple
                or any(not isinstance(value, C2RBCycleResidual)
                       for value in self.cycle_residuals)
                or type(self.failing_chords) is not tuple
                or tuple(sorted(set(self.failing_chords), key=_identity_key))
                   != self.failing_chords
                or any(value not in self.chords for value in self.failing_chords)
                or type(self.ambiguity_class_count_lower_bound) is not int
                or type(self.exhaustive) is not bool):
            raise ValueError("C2-R-B component result is malformed")
        if (len(self.tree_edges) != len(self.frames) - 1
                or any(identity.left_frame not in self.frames
                       or identity.right_frame not in self.frames
                       for identity in (*self.tree_edges, *self.chords))
                or any(tuple(frame for frame, _xyz in certificate.translations_dyadic_xyz)
                       != self.frames
                       or tuple(sorted(
                           (reference.match_identity for reference
                            in certificate.edge_references), key=_identity_key))
                       != tuple(sorted((*self.tree_edges, *self.chords), key=_identity_key))
                       for certificate in self.translation_classes)):
            raise ValueError("component records do not bind its complete canonical graph")
        if (len({value.translation_key for value in self.translation_classes})
                != len(self.translation_classes)):
            raise ValueError("component translation classes must have distinct keys")
        count = len(self.translation_classes)
        # ``cycle_residuals`` is outcome-specific: a contradiction records only
        # its failing chords (a proper subset when some cycles do close), while a
        # validated/ambiguous component records every chord. The branches below
        # are therefore the only place the residual length is bound.
        valid = (
            (self.outcome == "unique_validated" and self.cycle_rank > 0
             and count == 1 and self.exhaustive and not self.failing_chords
             and all(value.matching_provenance_count > 0
                     for value in self.cycle_residuals)
             and len(self.cycle_residuals) == self.cycle_rank
             and tuple(value.chord_identity for value in self.cycle_residuals)
                 == self.chords
             and self.ambiguity_class_count_lower_bound == 0)
            or (self.outcome == "unique_unvalidated_tree" and self.cycle_rank == 0
                and count == 1 and self.exhaustive and not self.failing_chords
                and not self.cycle_residuals
                and self.ambiguity_class_count_lower_bound == 0)
            or (self.outcome == "ambiguous" and count == 2
                and self.ambiguity_class_count_lower_bound == 2
                and not self.failing_chords
                and all(value.matching_provenance_count > 0
                        for value in self.cycle_residuals)
                and len(self.cycle_residuals) == self.cycle_rank
                and tuple(value.chord_identity for value in self.cycle_residuals)
                    == self.chords)
            or (self.outcome == "contradiction" and self.cycle_rank > 0
                and count == 0 and self.exhaustive and bool(self.failing_chords)
                and len(self.cycle_residuals) == len(self.failing_chords)
                and tuple(value.chord_identity for value in self.cycle_residuals)
                    == self.failing_chords
                and all(value.matching_provenance_count == 0
                        for value in self.cycle_residuals)
                and self.ambiguity_class_count_lower_bound == 0))
        if not valid:
            raise ValueError("component outcome does not match its proof records")
        if (not _is_digest(self.component_sha256)
                or _digest_json(_component_payload(self)) != self.component_sha256):
            raise ValueError("component digest does not bind its contents")

    def to_record(self) -> dict[str, Any]:
        record = _component_payload(self)
        record["component_sha256"] = self.component_sha256
        return record


def _result_payload(value: "C2RBResult") -> dict[str, Any]:
    return {
        "status": value.status,
        "placement_readiness": value.placement_readiness,
        "connected": value.connected,
        "vector_result_sha256": value.vector_result_sha256,
        "vector_input_sha256": value.vector_input_sha256,
        "comparison_tolerance": value.comparison_tolerance,
        "closure_tolerance": value.closure_tolerance,
        "budget": value.budget.to_record(),
        "deterministic_work": value.deterministic_work.to_record(),
        "runtime_work": value.runtime_work.to_record(),
        "components": [component.to_record() for component in value.components],
        "refusals": [refusal.to_record() for refusal in value.refusals],
        "budget_failure_reason": value.budget_failure_reason,
        "input_sha256": value.input_sha256,
    }


def _result_input_payload(value: "C2RBResult") -> dict[str, Any]:
    return {
        "vector_result_sha256": value.vector_result_sha256,
        "vector_input_sha256": value.vector_input_sha256,
        "comparison_tolerance": value.comparison_tolerance,
        "closure_tolerance": value.closure_tolerance,
        "budget": value.budget.to_record(),
    }


def _make_result(
    *, status: str, budget: C2RBudget,
    deterministic_work: C2RBDeterministicWork,
    runtime_work: C2RBRuntimeWork,
    vector_result_sha256: str, vector_input_sha256: str,
    comparison_tolerance: float | None, closure_tolerance: float | None,
    components: tuple[C2RBComponentResult, ...] = (),
    refusals: tuple[C2RBRefusal, ...] = (),
    budget_failure_reason: str | None = None,
) -> C2RBResult:
    connected = len(components) == 1
    if status == "complete":
        readiness = ("ready" if connected
                     and components[0].outcome == "unique_validated"
                     else "review_required")
    else:
        readiness = "unavailable"
    provisional = object.__new__(C2RBResult)
    values = {
        "status": status, "placement_readiness": readiness,
        "connected": connected,
        "vector_result_sha256": vector_result_sha256,
        "vector_input_sha256": vector_input_sha256,
        "comparison_tolerance": comparison_tolerance,
        "closure_tolerance": closure_tolerance,
        "budget": budget, "deterministic_work": deterministic_work,
        "runtime_work": runtime_work, "components": components,
        "refusals": _sort_refusals(refusals),
        "budget_failure_reason": budget_failure_reason,
    }
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    input_sha = _digest_json(_result_input_payload(provisional))
    object.__setattr__(provisional, "input_sha256", input_sha)
    object.__setattr__(provisional, "result_sha256", _digest_json(_result_payload(provisional)))
    C2RBResult.__post_init__(provisional)
    return provisional


def _deterministic_over(work: C2RBDeterministicWork,
                        budget: C2RBudget) -> bool:
    pairs = (
        ("components", "max_components"), ("frames", "max_frames"),
        ("edges", "max_edges"), ("tree_edges", "max_tree_edges"),
        ("chords", "max_chords"), ("cycle_rows", "max_cycle_rows"),
        ("p1d_entries", "max_p1d_entries"),
        ("residue_values", "max_residue_values"),
    )
    return any(getattr(work, counter) > getattr(budget, cap)
               for counter, cap in pairs)


def _runtime_over(work: C2RBRuntimeWork, budget: C2RBudget) -> bool:
    limits = _RuntimeCounter._limits
    return any(getattr(work, counter) > getattr(budget, cap)
               for counter, cap in limits.items())


@dataclass(frozen=True, slots=True)
class C2RBResult:
    status: str
    placement_readiness: str
    connected: bool
    vector_result_sha256: str
    vector_input_sha256: str
    comparison_tolerance: float | None
    closure_tolerance: float | None
    budget: C2RBudget
    deterministic_work: C2RBDeterministicWork
    runtime_work: C2RBRuntimeWork
    components: tuple[C2RBComponentResult, ...]
    refusals: tuple[C2RBRefusal, ...]
    budget_failure_reason: str | None
    input_sha256: str
    result_sha256: str

    def __post_init__(self) -> None:
        if (self.status not in _STATUS or self.placement_readiness not in _READINESS
                or type(self.connected) is not bool
                or not _is_digest(self.vector_result_sha256)
                or not _is_digest(self.vector_input_sha256)
                or not isinstance(self.budget, C2RBudget)
                or not isinstance(self.deterministic_work, C2RBDeterministicWork)
                or not isinstance(self.runtime_work, C2RBRuntimeWork)
                or type(self.components) is not tuple
                or any(not isinstance(value, C2RBComponentResult) for value in self.components)
                or type(self.refusals) is not tuple
                or any(not isinstance(value, C2RBRefusal) for value in self.refusals)):
            raise ValueError("C2-R-B result is malformed")
        for tolerance in (self.comparison_tolerance, self.closure_tolerance):
            if tolerance is not None and (
                    type(tolerance) not in (int, float)
                    or not math.isfinite(float(tolerance)) or tolerance < 0.0):
                raise ValueError("C2-R-B result tolerances must be finite and non-negative")
        if self.status == "complete":
            if (not self.components or self.refusals or self.budget_failure_reason is not None
                    or self.placement_readiness == "unavailable"
                    or self.connected != (len(self.components) == 1)):
                raise ValueError("complete C2-R-B result is not a full graph decision")
            expected_readiness = (
                "ready" if len(self.components) == 1
                and self.components[0].outcome == "unique_validated"
                else "review_required")
            if self.placement_readiness != expected_readiness:
                raise ValueError("placement readiness does not follow component decisions")
            if (_deterministic_over(self.deterministic_work, self.budget)
                    or _runtime_over(self.runtime_work, self.budget)):
                raise ValueError("complete result exceeds a declared whole-request budget")
            closure_value = _finite_float(self.closure_tolerance)
            if closure_value is None:
                raise ValueError(
                    "complete C2-R-B result requires a finite closure tolerance")
            closure_exact = Fraction.from_float(closure_value)
            for component in self.components:
                if component.outcome not in {"unique_validated", "ambiguous"}:
                    continue
                for cycle in component.cycle_residuals:
                    for low, high in cycle.residual_bounds_dyadic_xyz:
                        if (_fraction(low) > closure_exact
                                or _fraction(high) < -closure_exact):
                            raise ValueError(
                                "feasible cycle residual bounds exceed the closure tolerance")
        elif self.status == "refused":
            if (self.components or not self.refusals or self.placement_readiness != "unavailable"
                    or self.budget_failure_reason is not None
                    or self.deterministic_work
                        != C2RBDeterministicWork(0, 0, 0, 0, 0, 0, 0, 0, 0)
                    or self.runtime_work != C2RBRuntimeWork(0, 0, 0, 0, 0, 0, 0, 0)):
                raise ValueError("refused C2-R-B result must be atomic")
        else:
            if (self.components or self.refusals or self.placement_readiness != "unavailable"
                    or self.budget_failure_reason not in {
                        "planned_budget_exceeded", "runtime_budget_exceeded",
                        "runtime_count_mismatch"}):
                raise ValueError("budget-exhausted C2-R-B result must be atomic")
            if self.budget_failure_reason == "planned_budget_exceeded":
                if (not _deterministic_over(self.deterministic_work, self.budget)
                        or self.runtime_work != C2RBRuntimeWork(0, 0, 0, 0, 0, 0, 0, 0)):
                    raise ValueError("preflight exhaustion must publish exact work and zero runtime")
            elif _deterministic_over(self.deterministic_work, self.budget):
                raise ValueError("runtime exhaustion cannot publish a preflight-excess plan")
            if self.budget_failure_reason == "runtime_budget_exceeded" and not _runtime_over(
                    self.runtime_work, self.budget):
                raise ValueError("runtime exhaustion must expose its exceeded counter")
        if tuple(sorted(self.refusals, key=_ref_identity_key)) != self.refusals:
            raise ValueError("C2-R-B refusals must be deterministically ordered")
        if not _is_digest(self.input_sha256) or not _is_digest(self.result_sha256):
            raise ValueError("C2-R-B result hashes are malformed")
        if _digest_json(_result_input_payload(self)) != self.input_sha256:
            raise ValueError("C2-R-B input digest does not replay its request provenance")
        if _digest_json(_result_payload(self)) != self.result_sha256:
            raise ValueError("C2-R-B result digest does not replay")

    def to_record(self) -> dict[str, Any]:
        record = _result_payload(self)
        record["result_sha256"] = self.result_sha256
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return record

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.to_record(), sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")


@dataclass(frozen=True, slots=True)
class _TreeStep:
    identity: plane_axis_support.C1MatchIdentity
    source_frame: str
    target_frame: str
    direction: int


@dataclass(frozen=True, slots=True)
class _ComponentPlan:
    frames: tuple[str, ...]
    anchor: str
    tree_steps: tuple[_TreeStep, ...]
    tree_edges: tuple[plane_axis_support.C1MatchIdentity, ...]
    chords: tuple[plane_axis_support.C1MatchIdentity, ...]


@dataclass(frozen=True, slots=True)
class _GraphPlan:
    edges: tuple[plane_axis_support.C1MatchIdentity, ...]
    components: tuple[_ComponentPlan, ...]


@dataclass(frozen=True, slots=True)
class _SyntheticEdge:
    """Minimal private edge used by the exact singleton graph kernel."""

    identity: plane_axis_support.C1MatchIdentity
    delta: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


@dataclass(frozen=True, slots=True)
class _SingletonComponent:
    frames: tuple[str, ...]
    anchor: str
    translations: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]
    residuals: tuple[tuple[tuple[tuple[int, int], ...], ...], ...]
    outcome: str


def _make_graph_plan(
    identities: Sequence[plane_axis_support.C1MatchIdentity],
) -> _GraphPlan:
    """Canonical components, breadth-first trees, and fundamental cycles."""
    ordered = tuple(sorted(identities, key=_identity_key))
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("graph requires at least one unique edge identity")
    edge_by_id = {identity: identity for identity in ordered}
    adjacency: dict[str, list[tuple[plane_axis_support.C1MatchIdentity, str]]] = {}
    for identity in ordered:
        left, right = identity.left_frame, identity.right_frame
        if type(left) is not str or not left or type(right) is not str or not right or left == right:
            raise ValueError("edge endpoints must be distinct non-empty frame names")
        adjacency.setdefault(left, []).append((identity, right))
        adjacency.setdefault(right, []).append((identity, left))
    for neighbors in adjacency.values():
        neighbors.sort(key=lambda value: (_identity_key(value[0]), value[1]))

    components: list[_ComponentPlan] = []
    unseen = set(adjacency)
    tree_identity_set: set[plane_axis_support.C1MatchIdentity] = set()
    while unseen:
        anchor = min(unseen)
        parent: dict[str, tuple[str, plane_axis_support.C1MatchIdentity, int] | None] = {
            anchor: None}
        ordered_vertices = [anchor]
        queue = deque((anchor,))
        unseen.remove(anchor)
        tree_steps: list[_TreeStep] = []
        while queue:
            current = queue.popleft()
            for identity, neighbor in adjacency[current]:
                if neighbor in parent:
                    continue
                parent[neighbor] = (current, identity, 1 if current == identity.left_frame else -1)
                unseen.discard(neighbor)
                queue.append(neighbor)
                ordered_vertices.append(neighbor)
                tree_identity_set.add(identity)
                tree_steps.append(_TreeStep(
                    identity, current, neighbor,
                    1 if current == identity.left_frame else -1))
        frame_tuple = tuple(sorted(ordered_vertices))
        component_edge_ids = tuple(identity for identity in ordered
                                    if identity.left_frame in parent
                                    and identity.right_frame in parent)
        tree_ids = tuple(step.identity for step in tree_steps)
        chords = tuple(identity for identity in component_edge_ids
                       if identity not in set(tree_ids))

        components.append(_ComponentPlan(
            frame_tuple, anchor, tuple(tree_steps), tree_ids, chords))

    components.sort(key=lambda value: value.frames)
    return _GraphPlan(ordered, tuple(components))


def _single_delta_component(
    component: _ComponentPlan,
    edges: Mapping[plane_axis_support.C1MatchIdentity, _SyntheticEdge],
    tolerance: Fraction,
) -> _SingletonComponent:
    zero = (Fraction(0), Fraction(0), Fraction(0))
    translations: dict[str, tuple[Fraction, Fraction, Fraction]] = {
        component.anchor: zero}
    for step in component.tree_steps:
        source = translations[step.source_frame]
        delta = tuple(_fraction(value) for value in edges[step.identity].delta)
        translations[step.target_frame] = tuple(
            source[axis] + step.direction * delta[axis] for axis in range(3))  # type: ignore[assignment]
    residuals: list[tuple[tuple[int, int], ...]] = []
    closed = True
    for chord in component.chords:
        left_translation = translations[chord.left_frame]
        right_translation = translations[chord.right_frame]
        required = tuple(right_translation[axis] - left_translation[axis]
                         for axis in range(3))
        edge_delta = tuple(_fraction(value) for value in edges[chord].delta)
        residual = tuple(edge_delta[axis] - required[axis] for axis in range(3))
        dyadic = tuple(_dyadic(value) for value in residual)
        residuals.append(dyadic)
        if any(abs(value) > tolerance for value in residual):
            closed = False
    if not component.chords:
        outcome = "unique_unvalidated_tree"
    else:
        outcome = "unique_validated" if closed else "contradiction"
    return _SingletonComponent(
        component.frames, component.anchor,
        tuple((frame, tuple(_dyadic(value) for value in translations[frame]))
              for frame in component.frames), tuple(residuals), outcome)


def _singleton_graph_kernel(
    edges: Sequence[_SyntheticEdge], tolerance: Fraction,
) -> tuple[_GraphPlan, tuple[_SingletonComponent, ...]]:
    plan = _make_graph_plan(tuple(edge.identity for edge in edges))
    by_identity = {edge.identity: edge for edge in edges}
    if len(by_identity) != len(edges):
        raise ValueError("duplicate synthetic edge identity")
    solved = tuple(_single_delta_component(component, by_identity, tolerance)
                   for component in plan.components)
    return plan, solved


def _tree_cycle_participation(
    component: _ComponentPlan,
) -> dict[plane_axis_support.C1MatchIdentity, int]:
    """Count fundamental cycles crossing each tree edge without path lists."""
    parent = {component.anchor: component.anchor}
    depth = {component.anchor: 0}
    children: dict[str, list[str]] = {frame: [] for frame in component.frames}
    for step in component.tree_steps:
        parent[step.target_frame] = step.source_frame
        depth[step.target_frame] = depth[step.source_frame] + 1
        children[step.source_frame].append(step.target_frame)
    levels = max(1, len(component.frames).bit_length())
    ancestors: list[dict[str, str]] = [parent]
    for _level in range(1, levels):
        previous = ancestors[-1]
        ancestors.append({frame: previous[previous[frame]] for frame in component.frames})

    def lca(left: str, right: str) -> str:
        if depth[left] < depth[right]:
            left, right = right, left
        difference = depth[left] - depth[right]
        bit = 0
        while difference:
            if difference & 1:
                left = ancestors[bit][left]
            difference >>= 1
            bit += 1
        if left == right:
            return left
        for level in range(levels - 1, -1, -1):
            if ancestors[level][left] != ancestors[level][right]:
                left = ancestors[level][left]
                right = ancestors[level][right]
        return parent[left]

    difference = {frame: 0 for frame in component.frames}
    for chord in component.chords:
        common = lca(chord.left_frame, chord.right_frame)
        difference[chord.left_frame] += 1
        difference[chord.right_frame] += 1
        difference[common] -= 2
    result: dict[plane_axis_support.C1MatchIdentity, int] = {}
    for step in reversed(component.tree_steps):
        value = difference[step.target_frame]
        result[step.identity] = value
        difference[step.source_frame] += value
    return result


def _tree_order(
    component: _ComponentPlan,
    domains: Mapping[plane_axis_support.C1MatchIdentity,
                    plane_vector_domain.MatchLateralVectorDomain],
) -> tuple[plane_axis_support.C1MatchIdentity, ...]:
    participation = _tree_cycle_participation(component)
    return tuple(sorted(
        component.tree_edges,
        key=lambda identity: (-participation[identity],
                              domains[identity].represented_candidate_count,
                              _identity_key(identity))))


class _BudgetStop(Exception):
    def __init__(self, reason: str):
        self.reason = reason


class _IteratorMismatch(Exception):
    pass


class _RuntimeCounter:
    __slots__ = ("_budget", "_values")

    _limits = {
        "candidate_views_scanned": "max_candidate_views_scanned",
        "partial_assignments": "max_partial_assignments",
        "closure_component_checks": "max_closure_component_checks",
        "membership_probes": "max_membership_probes",
        "memoized_terms": "max_memoized_terms",
        "provenance_witnesses": "max_provenance_witnesses",
        "certificates_retained": "max_certificates",
    }

    def __init__(self, budget: C2RBudget):
        self._budget = budget
        self._values = {key: 0 for key in self._limits}

    def tick(self, field: str, amount: int = 1) -> None:
        if field not in self._limits or type(amount) is not int or amount < 0:
            raise _IteratorMismatch("invalid internal runtime counter update")
        next_value = self._values[field] + amount
        limit = getattr(self._budget, self._limits[field])
        self._values[field] = next_value
        if next_value > limit:
            raise _BudgetStop("runtime_budget_exceeded")

    def snapshot(self) -> C2RBRuntimeWork:
        return C2RBRuntimeWork(**self._values, candidate_at_calls=0)


@dataclass(frozen=True, slots=True)
class _DeltaClass:
    delta_dyadic_xyz: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    delta_xyz: tuple[float, float, float]
    multiplicity: int
    witness_index: int


@dataclass(frozen=True, slots=True)
class _Membership:
    count: int
    witness_index: int | None


@dataclass(frozen=True, slots=True)
class _PartialPotentials:
    values: Mapping[str, tuple[tuple[Fraction, Fraction], ...]]
    complete_path: frozenset[str]


def _seal_record(record_type: type, fields: Mapping[str, Any], digest_field: str,
                 payload_fn) -> Any:
    provisional = object.__new__(record_type)
    for name, value in fields.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, digest_field, _digest_json(payload_fn(provisional)))
    record_type.__post_init__(provisional)
    return provisional


def _sort_refusals(values: Iterable[C2RBRefusal]) -> tuple[C2RBRefusal, ...]:
    return tuple(sorted(values, key=_ref_identity_key))


def _dyadic_tuple(values: Sequence[float]) -> tuple[tuple[int, int], ...]:
    return tuple(_dyadic_from_float(value) for value in values)


def _delta_xyz(
    domain: plane_vector_domain.MatchLateralVectorDomain, index: int,
) -> tuple[float, float, float]:
    normal = domain.normal_constraint
    values: list[float | None] = [None, None, None]
    for axis, value in domain.vector_at(index):
        if type(axis) is not int or axis not in (0, 1, 2) or values[axis] is not None:
            raise _IteratorMismatch("A4 vector axes are duplicated or invalid")
        finite = _finite_float(value)
        if finite is None:
            raise _IteratorMismatch("A4 lateral delta is non-finite")
        values[axis] = finite
    normal_axis = normal.world_axis
    if values[normal_axis] is not None:
        raise _IteratorMismatch("A4 lateral delta repeats its normal axis")
    normal_value = _finite_float(normal.delta_translation)
    if normal_value is None:
        raise _IteratorMismatch("A4 normal delta is non-finite")
    values[normal_axis] = normal_value
    if any(value is None for value in values):
        raise _IteratorMismatch("A4 vector omits a world-axis delta")
    return tuple(values)  # type: ignore[return-value]


def _ref_from_domain(
    domain: plane_vector_domain.MatchLateralVectorDomain,
    index: int,
    *,
    tree_multiplicity: int | None = None,
    chord_count: int | None = None,
) -> C2RBEdgeCandidateRef:
    if (type(index) is not int or index < 0
            or index >= domain.represented_candidate_count):
        raise _IteratorMismatch("candidate reference index is outside its A4 domain")
    provenance = domain.provenance_key_at(index)
    if (type(provenance) is not tuple or len(provenance) != 2
            or type(provenance[0]) is not tuple or len(provenance[0]) != 2
            or type(provenance[1]) is not tuple or len(provenance[1]) != 2):
        raise _IteratorMismatch("A4 provenance key shape changed")
    rc = provenance[0]
    if (domain.joint_entry_count < 1
            or domain.represented_candidate_count % domain.joint_entry_count):
        raise _IteratorMismatch("A4 product dimensions do not resolve joint entry indices")
    per_entry = domain.represented_candidate_count // domain.joint_entry_count
    entry_index = index // per_entry
    if (entry_index >= len(domain.p1d_domain.entries)
            or (domain.p1d_domain.entries[entry_index].row_shift,
                domain.p1d_domain.entries[entry_index].column_shift) != rc):
        raise _IteratorMismatch("A4 provenance does not resolve one joint P1-D entry")
    residue_indices = []
    for factor, item in zip(domain.axis_factors, provenance[1]):
        if (type(item) is not tuple or len(item) != 5
                or item[0] != factor.world_axis
                or item[1] != factor.left_side.record_sha256
                or item[3] != factor.right_side.record_sha256
                or type(item[2]) is not int or type(item[4]) is not int
                or item[2] < 0 or item[2] >= factor.left_side.candidate_count
                or item[4] < 0 or item[4] >= factor.right_side.candidate_count):
            raise _IteratorMismatch("A4 residue provenance does not replay its factors")
        residue_indices.extend((item[2], item[4]))
    floats = _delta_xyz(domain, index)
    dyadics = _dyadic_tuple(floats)
    provisional_fields = {
        "match_identity": domain.match_identity,
        "domain_sha256": domain.domain_sha256,
        "global_index": index,
        "p1d_entry_index": entry_index,
        "shift_entry_rc": rc,
        "axis_worlds": tuple(factor.world_axis for factor in domain.axis_factors),
        "residue_side_sha256s": tuple(
            digest
            for factor in domain.axis_factors
            for digest in (factor.left_side.record_sha256,
                           factor.right_side.record_sha256)),
        "residue_candidate_counts": tuple(
            count
            for factor in domain.axis_factors
            for count in (factor.left_side.candidate_count,
                          factor.right_side.candidate_count)),
        "p1d_entry_count": domain.joint_entry_count,
        "p1d_max_shift_cells": domain.p1d_domain.max_shift_cells,
        "residue_indices": tuple(residue_indices),
        "provenance_key": provenance,
        "delta_dyadic_xyz": dyadics,
        "delta_xyz": floats,
        "tree_delta_class_multiplicity": tree_multiplicity,
        "chord_satisfying_provenance_count": chord_count,
    }
    return _seal_record(C2RBEdgeCandidateRef, provisional_fields, "ref_sha256",
                        _edge_ref_payload)


def _interval_add(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction],
    sign: int = 1,
) -> tuple[Fraction, Fraction]:
    if sign == 1:
        return left[0] + right[0], left[1] + right[1]
    return left[0] - right[1], left[1] - right[0]


def _domain_bounds(
    domain: plane_vector_domain.MatchLateralVectorDomain,
    runtime: _RuntimeCounter,
) -> tuple[tuple[Fraction, Fraction], ...]:
    """Compute safe axis bounds from indexed entries and residue endpoints."""
    lows: list[Fraction | None] = [None, None, None]
    highs: list[Fraction | None] = [None, None, None]
    normal = domain.normal_constraint
    normal_axis = normal.world_axis
    normal_value = _fraction(_dyadic_from_float(float(normal.delta_translation)))
    lows[normal_axis] = highs[normal_axis] = normal_value
    expected_start = 0
    vectors_per_entry = (domain.represented_candidate_count // domain.joint_entry_count)
    for entry_index, (entry, start, stop) in enumerate(domain.iter_entry_groups()):
        if (type(start) is not int or type(stop) is not int or start != expected_start
                or stop != start + vectors_per_entry
                or entry_index >= len(domain.p1d_domain.entries)
                or entry != domain.p1d_domain.entries[entry_index]):
            raise _IteratorMismatch("A4 entry groups are not the canonical complete partition")
        expected_start = stop
        for factor in domain.axis_factors:
            shift = (entry.column_shift if factor.right_shift_source_component == "column"
                     else entry.row_shift)
            left_values = (factor.left_side.candidate_values[0],
                           factor.left_side.candidate_values[-1])
            right_values = (factor.right_side.candidate_values[0],
                            factor.right_side.candidate_values[-1])
            axis_values: list[Fraction] = []
            for left_value in dict.fromkeys(left_values):
                for right_value in dict.fromkeys(right_values):
                    runtime.tick("membership_probes")
                    value = (factor.right_world_sign * shift
                             + factor.left_world_sign * left_value
                             - factor.right_world_sign * right_value)
                    finite = _finite_float(value)
                    if finite is None:
                        raise _IteratorMismatch("A4 factor bound became non-finite")
                    axis_values.append(_fraction(_dyadic_from_float(finite)))
            axis = factor.world_axis
            local_low, local_high = min(axis_values), max(axis_values)
            lows[axis] = local_low if lows[axis] is None else min(lows[axis], local_low)
            highs[axis] = local_high if highs[axis] is None else max(highs[axis], local_high)
    if expected_start != domain.represented_candidate_count or any(
            value is None for value in (*lows, *highs)):
        raise _IteratorMismatch("A4 entry groups or world-axis bounds are incomplete")
    return tuple((lows[axis], highs[axis]) for axis in range(3))  # type: ignore[return-value]


def _delta_classes(
    domain: plane_vector_domain.MatchLateralVectorDomain,
    runtime: _RuntimeCounter,
) -> tuple[_DeltaClass, ...]:
    classes: dict[tuple[tuple[int, int], ...], list[Any]] = {}
    expected_index = 0
    iterator = domain.iter_views(0, domain.represented_candidate_count)
    for view in iterator:
        if (not isinstance(view, plane_vector_domain.LateralVectorView)
                or view.candidate_index != expected_index):
            raise _IteratorMismatch("A4 compact view iterator skipped or reordered an index")
        runtime.tick("candidate_views_scanned")
        values: list[float | None] = [None, None, None]
        for axis, raw in view.vector:
            if type(axis) is not int or axis not in (0, 1, 2) or values[axis] is not None:
                raise _IteratorMismatch("A4 compact view has invalid world axes")
            values[axis] = _finite_float(raw)
        normal_axis = domain.normal_constraint.world_axis
        values[normal_axis] = _finite_float(domain.normal_constraint.delta_translation)
        if any(value is None for value in values):
            raise _IteratorMismatch("A4 compact view is missing a finite axis delta")
        floats = tuple(values)  # type: ignore[assignment]
        dyadics = _dyadic_tuple(floats)
        prior = classes.get(dyadics)
        if prior is None:
            runtime.tick("memoized_terms")
            classes[dyadics] = [1, expected_index, floats]
        else:
            prior[0] += 1
        expected_index += 1
    if expected_index != domain.represented_candidate_count:
        raise _IteratorMismatch("A4 compact view iterator ended before its declared product")
    return tuple(_DeltaClass(key, data[2], data[0], data[1])
                 for key, data in classes.items())


def _factorized_single_delta_class(
    domain: plane_vector_domain.MatchLateralVectorDomain,
    runtime: _RuntimeCounter,
) -> _DeltaClass | None:
    """Prove one exact complete delta from factor extrema, without leaf scans.

    Each lateral factor is monotone in both sorted residue tables. The sign of
    each coefficient selects its exact low/high endpoint. Evaluating those two
    corners in A4's frozen binary64 order bounds every represented leaf. Equal
    rounded extrema across every P1-D entry prove that the whole axis factor is
    one exact stored float; no joint candidate iterator is needed.

    This endpoint proof runs for every represented count, including one. There
    is deliberately no ``represented_candidate_count == 1`` shortcut: the public
    N=1 fixture must execute the factorized proof itself.

    A represented count greater than one cannot pass this proof under the
    current A4 invariants, so no such fixture is invented. ``ResidueSideTable``
    requires strictly increasing ``candidate_values`` whose consecutive step is
    the P1-B period grid step rather than one binary64 ULP, and the signed
    residue coefficient is never zero. Holding the other table fixed therefore
    gives at least two distinct rounded lateral values whenever a factor has
    more than one residue pair. When every factor has exactly one residue pair,
    ``joint_entry_count`` distinct row-major P1-D entries still change one
    row-sourced and one column-sourced integer shift, so each entry again yields
    a distinct lateral value. A single exact numeric delta therefore implies
    ``represented_candidate_count == 1``; the factorized proof remains the
    mandatory eligibility mechanism and is tested on the public N=1 case rather
    than bypassed.
    """
    normal = domain.normal_constraint
    normal_axis = normal.world_axis
    normal_value = _finite_float(normal.delta_translation)
    if (type(normal_axis) is not int or normal_axis not in (0, 1, 2)
            or normal_value is None):
        raise _IteratorMismatch("A4 normal factor is not one finite world-axis value")
    factors = tuple(domain.axis_factors)
    if (len(factors) != 2
            or {factor.world_axis for factor in factors}
                != set(range(3)) - {normal_axis}):
        raise _IteratorMismatch("A4 lateral factors do not partition the normal axis")
    joint_count = domain.joint_entry_count
    represented = domain.represented_candidate_count
    if (type(joint_count) is not int or joint_count < 1
            or represented % joint_count):
        raise _IteratorMismatch("A4 product dimensions cannot resolve entry groups")
    vectors_per_entry = represented // joint_count
    extrema_by_axis: dict[int, tuple[float, float]] = {}
    expected_start = 0
    for entry_index, (entry, start, stop) in enumerate(domain.iter_entry_groups()):
        if (type(start) is not int or type(stop) is not int
                or start != expected_start or stop != start + vectors_per_entry
                or entry_index >= len(domain.p1d_domain.entries)
                or entry != domain.p1d_domain.entries[entry_index]):
            raise _IteratorMismatch("A4 entry groups are not a complete canonical partition")
        expected_start = stop
        for factor in factors:
            if (type(factor.world_axis) is not int or factor.world_axis not in (0, 1, 2)
                    or factor.world_axis == normal_axis
                    or factor.left_world_sign not in (-1, 1)
                    or factor.right_world_sign not in (-1, 1)
                    or factor.right_shift_source_component not in ("row", "column")):
                raise _IteratorMismatch("A4 lateral factor signs/axes are invalid")
            left_values = factor.left_side.candidate_values
            right_values = factor.right_side.candidate_values
            if (not left_values or not right_values
                    or factor.left_side.candidate_count != len(left_values)
                    or factor.right_side.candidate_count != len(right_values)):
                raise _IteratorMismatch("A4 residue endpoints are unavailable")
            left_low, left_high = left_values[0], left_values[-1]
            right_low, right_high = right_values[0], right_values[-1]
            left_for_min = left_low if factor.left_world_sign > 0 else left_high
            left_for_max = left_high if factor.left_world_sign > 0 else left_low
            right_for_min = right_high if factor.right_world_sign > 0 else right_low
            right_for_max = right_low if factor.right_world_sign > 0 else right_high
            shift = (entry.column_shift
                     if factor.right_shift_source_component == "column"
                     else entry.row_shift)
            # Keep A4's exact evaluation order: shift term, then left term,
            # then subtraction of the signed right residue.
            low_raw = (factor.right_world_sign * shift
                       + factor.left_world_sign * left_for_min
                       - factor.right_world_sign * right_for_min)
            high_raw = (factor.right_world_sign * shift
                        + factor.left_world_sign * left_for_max
                        - factor.right_world_sign * right_for_max)
            low = _finite_float(low_raw)
            high = _finite_float(high_raw)
            runtime.tick("membership_probes", 2)
            if low is None or high is None:
                raise _IteratorMismatch("A4 endpoint corner became non-finite")
            prior = extrema_by_axis.get(factor.world_axis)
            if prior is not None:
                low = min(low, prior[0])
                high = max(high, prior[1])
            extrema_by_axis[factor.world_axis] = (low, high)
            if low != high:
                return None
    if expected_start != represented or set(extrema_by_axis) != set(range(3)) - {normal_axis}:
        raise _IteratorMismatch("A4 factor groups omit entries or world axes")
    if any(low != high for low, high in extrema_by_axis.values()):
        return None
    values: list[float | None] = [None, None, None]
    values[normal_axis] = normal_value
    for axis, (low, _high) in extrema_by_axis.items():
        values[axis] = low
    if any(value is None for value in values):
        raise _IteratorMismatch("A4 exact singleton proof omitted a world-axis delta")
    floats = tuple(values)  # type: ignore[assignment]
    return _DeltaClass(_dyadic_tuple(floats), floats, represented, 0)


def _factorized_membership(
    domain: plane_vector_domain.MatchLateralVectorDomain,
    required: tuple[Fraction, Fraction, Fraction],
    tolerance: Fraction,
    runtime: _RuntimeCounter,
) -> _Membership:
    """Count chord provenance by one-entry/two-axis residue-pair factoring."""
    normal = domain.normal_constraint
    normal_actual = _fraction(_dyadic_from_float(float(normal.delta_translation)))
    normal_matches = abs(normal_actual - required[normal.world_axis]) <= tolerance
    factor0, factor1 = domain.axis_factors
    pair_counts = tuple(factor.left_side.candidate_count * factor.right_side.candidate_count
                        for factor in domain.axis_factors)
    per_entry = pair_counts[0] * pair_counts[1]
    expected_start = 0
    total = 0
    first_global_index: int | None = None
    for entry_index, (entry, start, stop) in enumerate(domain.iter_entry_groups()):
        if (start != expected_start or stop != start + per_entry
                or entry_index >= len(domain.p1d_domain.entries)
                or entry != domain.p1d_domain.entries[entry_index]):
            raise _IteratorMismatch("A4 chord entry groups do not replay the mixed-radix index")
        expected_start = stop
        axis_counts: list[int] = []
        first_pairs: list[tuple[int, int] | None] = []
        for factor in domain.axis_factors:
            shift = (entry.column_shift if factor.right_shift_source_component == "column"
                     else entry.row_shift)
            matches = 0
            first_pair = None
            for left_index, left_value in enumerate(factor.left_side.candidate_values):
                for right_index, right_value in enumerate(factor.right_side.candidate_values):
                    runtime.tick("membership_probes")
                    # Preserve A4's exact left-to-right binary64 evaluation order.
                    raw = (factor.right_world_sign * shift
                           + factor.left_world_sign * left_value
                           - factor.right_world_sign * right_value)
                    finite = _finite_float(raw)
                    if finite is None:
                        raise _IteratorMismatch("A4 chord factor produced a non-finite value")
                    if abs(_fraction(_dyadic_from_float(finite))
                           - required[factor.world_axis]) <= tolerance:
                        matches += 1
                        if first_pair is None:
                            first_pair = (left_index, right_index)
            axis_counts.append(matches)
            first_pairs.append(first_pair)
        entry_count = axis_counts[0] * axis_counts[1]
        total += entry_count
        if entry_count and first_global_index is None:
            pair0, pair1 = first_pairs
            assert pair0 is not None and pair1 is not None
            q0 = pair0[0] * factor0.right_side.candidate_count + pair0[1]
            q1 = pair1[0] * factor1.right_side.candidate_count + pair1[1]
            first_global_index = start + q0 * pair_counts[1] + q1
    if expected_start != domain.represented_candidate_count:
        raise _IteratorMismatch("A4 chord entry table does not cover its represented product")
    if not normal_matches:
        return _Membership(0, None)
    return _Membership(total, first_global_index)


def _partial_potentials(
    component: _ComponentPlan,
    assignments: Mapping[plane_axis_support.C1MatchIdentity, _DeltaClass],
    bounds: Mapping[plane_axis_support.C1MatchIdentity,
                    tuple[tuple[Fraction, Fraction], ...]],
) -> _PartialPotentials:
    zero = ((Fraction(0), Fraction(0)),) * 3
    values: dict[str, tuple[tuple[Fraction, Fraction], ...]] = {component.anchor: zero}
    complete = {component.anchor}
    for step in component.tree_steps:
        source = values.get(step.source_frame)
        if source is None:
            continue
        assigned = assignments.get(step.identity)
        edge_bounds = (tuple((value, value) for value in
                              (_fraction(item) for item in assigned.delta_dyadic_xyz))
                       if assigned is not None else bounds[step.identity])
        values[step.target_frame] = tuple(
            _interval_add(source[axis], edge_bounds[axis], step.direction)
            for axis in range(3))
        if step.source_frame in complete and assigned is not None:
            complete.add(step.target_frame)
    return _PartialPotentials(values, frozenset(complete))


def _cycle_interval_can_close(
    chord: plane_axis_support.C1MatchIdentity,
    potentials: _PartialPotentials,
    bounds: Mapping[plane_axis_support.C1MatchIdentity,
                    tuple[tuple[Fraction, Fraction], ...]],
    tolerance: Fraction,
) -> bool:
    left = potentials.values[chord.left_frame]
    right = potentials.values[chord.right_frame]
    chord_bounds = bounds[chord]
    for axis in range(3):
        tree_difference = (right[axis][0] - left[axis][1],
                           right[axis][1] - left[axis][0])
        residual = _interval_add(chord_bounds[axis], tree_difference, -1)
        if residual[0] > tolerance or residual[1] < -tolerance:
            return False
    return True


def _cycle_record(
    chord: plane_axis_support.C1MatchIdentity,
    required: tuple[Fraction, Fraction, Fraction],
    chord_bounds: tuple[tuple[Fraction, Fraction], ...],
    match_count: int,
) -> C2RBCycleResidual:
    required_dyadics = tuple(_dyadic(value) for value in required)
    residual_bounds = tuple(
        (_dyadic(chord_bounds[axis][0] - required[axis]),
         _dyadic(chord_bounds[axis][1] - required[axis]))
        for axis in range(3))
    fields = {
        "chord_identity": chord,
        "required_chord_delta_dyadic_xyz": required_dyadics,
        "residual_bounds_dyadic_xyz": residual_bounds,
        "matching_provenance_count": match_count,
    }
    return _seal_record(C2RBCycleResidual, fields, "cycle_sha256", _cycle_payload)


@dataclass(frozen=True, slots=True)
class _FeasibleAssignment:
    translation_key: tuple[Any, ...]
    translations: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]
    tree_classes: Mapping[plane_axis_support.C1MatchIdentity, _DeltaClass]
    chord_memberships: Mapping[plane_axis_support.C1MatchIdentity, _Membership]
    cycle_records: tuple[C2RBCycleResidual, ...]
    multiplicity: int


def _make_component_result(fields: Mapping[str, Any]) -> C2RBComponentResult:
    return _seal_record(C2RBComponentResult, fields, "component_sha256", _component_payload)


def _make_certificate(
    component: _ComponentPlan,
    domains: Mapping[plane_axis_support.C1MatchIdentity,
                     plane_vector_domain.MatchLateralVectorDomain],
    feasible: _FeasibleAssignment,
    runtime: _RuntimeCounter,
) -> C2RBTranslationClassCertificate:
    runtime.tick("certificates_retained")
    references: list[C2RBEdgeCandidateRef] = []
    tree_set = set(component.tree_edges)
    for identity in sorted((*component.tree_edges, *component.chords), key=_identity_key):
        runtime.tick("provenance_witnesses")
        if identity in tree_set:
            delta_class = feasible.tree_classes[identity]
            reference = _ref_from_domain(
                domains[identity], delta_class.witness_index,
                tree_multiplicity=delta_class.multiplicity)
        else:
            membership = feasible.chord_memberships[identity]
            if membership.count < 1 or membership.witness_index is None:
                raise _IteratorMismatch("feasible class lacks a chord witness")
            reference = _ref_from_domain(
                domains[identity], membership.witness_index,
                chord_count=membership.count)
        references.append(reference)
    return _seal_record(C2RBTranslationClassCertificate, {
        "gauge_anchor": component.anchor,
        "translations_dyadic_xyz": feasible.translations,
        "edge_references": tuple(references),
        "assignment_provenance_multiplicity": feasible.multiplicity,
    }, "certificate_sha256", _certificate_payload)


def _make_component_output(
    component: _ComponentPlan,
    domains: Mapping[plane_axis_support.C1MatchIdentity,
                     plane_vector_domain.MatchLateralVectorDomain],
    outcome: str,
    assignments: Sequence[_FeasibleAssignment],
    failure_cycles: Mapping[plane_axis_support.C1MatchIdentity, C2RBCycleResidual],
    exhaustive: bool,
    runtime: _RuntimeCounter,
) -> C2RBComponentResult:
    certificates = tuple(_make_certificate(component, domains, item, runtime)
                         for item in assignments)
    if outcome == "contradiction":
        failed = tuple(sorted(failure_cycles, key=_identity_key))
        cycles = tuple(failure_cycles[identity] for identity in failed)
    elif assignments:
        failed = ()
        cycles = assignments[0].cycle_records
    else:
        failed = ()
        cycles = ()
    return _make_component_result({
        "frames": component.frames, "gauge_anchor": component.anchor,
        "tree_edges": component.tree_edges, "chords": component.chords,
        "cycle_rank": len(component.chords), "outcome": outcome,
        "translation_classes": certificates, "cycle_residuals": cycles,
        "failing_chords": failed,
        "ambiguity_class_count_lower_bound": 2 if outcome == "ambiguous" else 0,
        "exhaustive": exhaustive,
    })


def _deterministic_work(
    plan: _GraphPlan,
    domains: Mapping[plane_axis_support.C1MatchIdentity,
                     plane_vector_domain.MatchLateralVectorDomain],
) -> C2RBDeterministicWork:
    edges = len(plan.edges)
    components = len(plan.components)
    tree_edges = sum(len(value.tree_edges) for value in plan.components)
    chords = sum(len(value.chords) for value in plan.components)
    frames = sum(len(value.frames) for value in plan.components)
    return C2RBDeterministicWork(
        components, frames, edges, tree_edges, chords, 3 * chords,
        sum(domain.joint_entry_count for domain in domains.values()),
        sum(side.candidate_count for domain in domains.values()
            for factor in domain.axis_factors
            for side in (factor.left_side, factor.right_side)),
        sum(domain.represented_candidate_count for domain in domains.values()))


def _input_refusal(reason: str, detail: str,
                   identity: plane_axis_support.C1MatchIdentity | None = None
                   ) -> C2RBRefusal:
    return C2RBRefusal(reason, detail[:240] or reason, identity)


def _refused_result(
    vector_result: Any,
    budget: C2RBudget,
    comparison_tolerance: Any,
    closure_tolerance: Any,
    refusal: C2RBRefusal,
) -> C2RBResult:
    result_sha = getattr(vector_result, "result_sha256", None)
    input_sha = getattr(vector_result, "input_sha256", None)
    if not _is_digest(result_sha):
        result_sha = "0" * 64
    if not _is_digest(input_sha):
        input_sha = "0" * 64
    comparison = (_finite_float(comparison_tolerance)
                  if type(comparison_tolerance) in (int, float) else None)
    closure = (_finite_float(closure_tolerance)
               if type(closure_tolerance) in (int, float) else None)
    if comparison is not None and comparison < 0.0:
        comparison = None
    if closure is not None and closure < 0.0:
        closure = None
    zero_work = C2RBDeterministicWork(0, 0, 0, 0, 0, 0, 0, 0, 0)
    return _make_result(
        status="refused", budget=budget, deterministic_work=zero_work,
        runtime_work=C2RBRuntimeWork(0, 0, 0, 0, 0, 0, 0, 0),
        vector_result_sha256=result_sha, vector_input_sha256=input_sha,
        comparison_tolerance=comparison, closure_tolerance=closure,
        refusals=(refusal,))


def _budget_exceeded_result(
    vector_result: plane_vector_domain.C2RVectorDomainResult,
    budget: C2RBudget,
    comparison_tolerance: float,
    closure_tolerance: float,
    deterministic: C2RBDeterministicWork,
    runtime: C2RBRuntimeWork,
    reason: str,
) -> C2RBResult:
    return _make_result(
        status="budget_exhausted", budget=budget,
        deterministic_work=deterministic, runtime_work=runtime,
        vector_result_sha256=vector_result.result_sha256,
        vector_input_sha256=vector_result.input_sha256,
        comparison_tolerance=comparison_tolerance,
        closure_tolerance=closure_tolerance,
        budget_failure_reason=reason)


def _preflight_budget_exceeds(
    work: C2RBDeterministicWork, budget: C2RBudget,
) -> bool:
    return _deterministic_over(work, budget)


def _translations_from_assignment(
    component: _ComponentPlan,
    assignments: Mapping[plane_axis_support.C1MatchIdentity, _DeltaClass],
) -> dict[str, tuple[Fraction, Fraction, Fraction]]:
    zero = (Fraction(0), Fraction(0), Fraction(0))
    translations: dict[str, tuple[Fraction, Fraction, Fraction]] = {
        component.anchor: zero}
    for step in component.tree_steps:
        if step.identity not in assignments:
            continue
        source = translations.get(step.source_frame)
        if source is None:
            continue
        delta = tuple(_fraction(value)
                      for value in assignments[step.identity].delta_dyadic_xyz)
        translations[step.target_frame] = tuple(
            source[axis] + step.direction * delta[axis] for axis in range(3))
    return translations


def _translation_records(
    component: _ComponentPlan,
    translations: Mapping[str, tuple[Fraction, Fraction, Fraction]],
) -> tuple[tuple[str, tuple[tuple[int, int], ...]], ...]:
    if tuple(sorted(translations)) != component.frames:
        raise _IteratorMismatch("complete tree assignment did not propagate to every frame")
    return tuple((frame, tuple(_dyadic(value) for value in translations[frame]))
                 for frame in component.frames)


def _point_bounds(
    domain: plane_vector_domain.MatchLateralVectorDomain, index: int,
) -> tuple[tuple[Fraction, Fraction], ...]:
    values = tuple(_fraction(value) for value in _dyadic_tuple(_delta_xyz(domain, index)))
    return tuple((value, value) for value in values)


def _membership_for_chord(
    domain: plane_vector_domain.MatchLateralVectorDomain,
    required: tuple[Fraction, Fraction, Fraction],
    tolerance: Fraction,
    runtime: _RuntimeCounter,
    singleton_class: _DeltaClass | None = None,
) -> _Membership:
    if singleton_class is not None:
        actual = tuple(_fraction(value)
                       for value in singleton_class.delta_dyadic_xyz)
        matches = all(abs(actual[axis] - required[axis]) <= tolerance
                      for axis in range(3))
        return _Membership(singleton_class.multiplicity if matches else 0,
                           0 if matches else None)
    return _factorized_membership(domain, required, tolerance, runtime)


def _evaluate_tree_assignment(
    component: _ComponentPlan,
    domains: Mapping[plane_axis_support.C1MatchIdentity,
                     plane_vector_domain.MatchLateralVectorDomain],
    tree_classes: Mapping[plane_axis_support.C1MatchIdentity, _DeltaClass],
    bounds: Mapping[plane_axis_support.C1MatchIdentity,
                    tuple[tuple[Fraction, Fraction], ...]],
    tolerance: Fraction,
    runtime: _RuntimeCounter,
    singleton_classes: Mapping[
        plane_axis_support.C1MatchIdentity, _DeltaClass] | None = None,
) -> tuple[_FeasibleAssignment | None,
           dict[plane_axis_support.C1MatchIdentity, C2RBCycleResidual]]:
    translations = _translations_from_assignment(component, tree_classes)
    if tuple(sorted(translations)) != component.frames:
        raise _IteratorMismatch("complete tree assignment omitted a frame")
    memberships: dict[plane_axis_support.C1MatchIdentity, _Membership] = {}
    cycle_records: list[C2RBCycleResidual] = []
    failed: dict[plane_axis_support.C1MatchIdentity, C2RBCycleResidual] = {}
    for chord in component.chords:
        runtime.tick("closure_component_checks")
        left = translations[chord.left_frame]
        right = translations[chord.right_frame]
        required = tuple(right[axis] - left[axis] for axis in range(3))
        membership = _membership_for_chord(
            domains[chord], required, tolerance, runtime,
            None if singleton_classes is None else singleton_classes.get(chord))
        memberships[chord] = membership
        cycle = _cycle_record(chord, required, bounds[chord], membership.count)
        cycle_records.append(cycle)
        if membership.count == 0:
            failed[chord] = cycle
    if failed:
        return None, failed
    translations_record = _translation_records(component, translations)
    multiplicity = math.prod(delta_class.multiplicity
                             for delta_class in tree_classes.values())
    multiplicity *= math.prod(membership.count for membership in memberships.values())
    feasible = _FeasibleAssignment(
        translations_record, translations_record, dict(tree_classes), memberships,
        tuple(cycle_records), multiplicity)
    return feasible, {}


def _singleton_component_result(
    component: _ComponentPlan,
    domains: Mapping[plane_axis_support.C1MatchIdentity,
                     plane_vector_domain.MatchLateralVectorDomain],
    singleton_classes: Mapping[
        plane_axis_support.C1MatchIdentity, _DeltaClass],
    tolerance: Fraction,
    runtime: _RuntimeCounter,
) -> C2RBComponentResult:
    tree_classes = {identity: singleton_classes[identity]
                    for identity in component.tree_edges}
    bounds = {identity: _point_bounds(domains[identity], 0)
              for identity in component.chords}
    feasible, failed = _evaluate_tree_assignment(
        component, domains, tree_classes, bounds, tolerance, runtime,
        singleton_classes)
    if failed:
        return _make_component_output(
            component, domains, "contradiction", (), failed, True, runtime)
    assert feasible is not None
    outcome = "unique_validated" if component.chords else "unique_unvalidated_tree"
    return _make_component_output(
        component, domains, outcome, (feasible,), {}, True, runtime)


def _solve_general_component(
    component: _ComponentPlan,
    domains: Mapping[plane_axis_support.C1MatchIdentity,
                     plane_vector_domain.MatchLateralVectorDomain],
    tolerance: Fraction,
    runtime: _RuntimeCounter,
) -> C2RBComponentResult:
    bounds = {identity: _domain_bounds(domains[identity], runtime)
              for identity in (*component.tree_edges, *component.chords)}
    variable_order = _tree_order(component, domains)
    groups = {identity: _delta_classes(domains[identity], runtime)
              for identity in variable_order}
    if not variable_order or any(not groups[identity] for identity in variable_order):
        raise _IteratorMismatch("a connected component has no spanning-tree domain values")

    if all(len(groups[identity]) == 1 for identity in variable_order):
        assignment = {identity: groups[identity][0] for identity in variable_order}
        feasible, failed = _evaluate_tree_assignment(
            component, domains, assignment, bounds, tolerance, runtime)
        if failed:
            return _make_component_output(
                component, domains, "contradiction", (), failed, True, runtime)
        assert feasible is not None
        outcome = "unique_validated" if component.chords else "unique_unvalidated_tree"
        return _make_component_output(
            component, domains, outcome, (feasible,), {}, True, runtime)

    solutions: dict[tuple[Any, ...], _FeasibleAssignment] = {}
    failures: dict[plane_axis_support.C1MatchIdentity, C2RBCycleResidual] = {}
    assignments: dict[plane_axis_support.C1MatchIdentity, _DeltaClass] = {}
    stopped_after_ambiguity = False

    def search(index: int) -> None:
        nonlocal stopped_after_ambiguity
        if stopped_after_ambiguity:
            return
        if index == len(variable_order):
            feasible, failed = _evaluate_tree_assignment(
                component, domains, assignments, bounds, tolerance, runtime)
            for identity, cycle in failed.items():
                failures.setdefault(identity, cycle)
            if feasible is None:
                return
            existing = solutions.get(feasible.translation_key)
            if existing is None:
                solutions[feasible.translation_key] = feasible
            else:
                solutions[feasible.translation_key] = _FeasibleAssignment(
                    existing.translation_key, existing.translations,
                    existing.tree_classes, existing.chord_memberships,
                    existing.cycle_records,
                    existing.multiplicity + feasible.multiplicity)
            if len(solutions) >= 2:
                stopped_after_ambiguity = True
            return

        identity = variable_order[index]
        for delta_class in groups[identity]:
            if stopped_after_ambiguity:
                return
            assignments[identity] = delta_class
            runtime.tick("partial_assignments")
            # Every chord reads the one global tree assignment.  A conservative
            # exact interval may reject it only after both endpoint paths are
            # fixed; every axis participates in that proof.
            potentials = _partial_potentials(component, assignments, bounds)
            impossible = False
            for chord in component.chords:
                if (chord.left_frame not in potentials.complete_path
                        or chord.right_frame not in potentials.complete_path):
                    continue
                runtime.tick("closure_component_checks")
                if not _cycle_interval_can_close(chord, potentials, bounds, tolerance):
                    required = tuple(
                        potentials.values[chord.right_frame][axis][0]
                        - potentials.values[chord.left_frame][axis][0]
                        for axis in range(3))
                    failures.setdefault(
                        chord, _cycle_record(chord, required, bounds[chord], 0))
                    impossible = True
            if not impossible:
                search(index + 1)
            del assignments[identity]

    search(0)
    if len(solutions) >= 2:
        retained = tuple(list(solutions.values())[:2])
        return _make_component_output(
            component, domains, "ambiguous", retained, {},
            not stopped_after_ambiguity, runtime)
    if solutions:
        return _make_component_output(
            component, domains, "unique_validated" if component.chords
            else "unique_unvalidated_tree", tuple(solutions.values()), {}, True, runtime)
    return _make_component_output(
        component, domains, "contradiction", (), failures, True, runtime)


def solve_relative_translation_graph(
    vector_result: plane_vector_domain.C2RVectorDomainResult,
    support: plane_axis_support.C1AxisSupport,
    p1d_result: plane_shift_domain.ShiftDomainResult,
    matches: Sequence[plane_graph.PlaneMatch],
    phase_evidence: Sequence[plane_residue.HypothesisPhaseShiftEvidence],
    *,
    readings,
    signatures,
    comparison_tolerance: float,
    closure_tolerance: float,
    budget: C2RBudget,
) -> C2RBResult:
    """Solve the exact directed relative-translation graph under explicit caps."""
    if not isinstance(budget, C2RBudget):
        raise TypeError("solve_relative_translation_graph requires C2RBudget")

    comparison = _finite_float(comparison_tolerance)
    closure = _finite_float(closure_tolerance)
    if comparison is None or comparison < 0.0:
        return _refused_result(
            vector_result, budget, comparison_tolerance, closure_tolerance,
            _input_refusal("comparison_tolerance_mismatch",
                            "comparison tolerance is not a finite numeric value"))
    if closure is None or closure < 0.0:
        return _refused_result(
            vector_result, budget, comparison, closure_tolerance,
            _input_refusal("closure_tolerance_invalid",
                            "closure tolerance must be finite and non-negative"))
    if not isinstance(vector_result, plane_vector_domain.C2RVectorDomainResult):
        return _refused_result(
            vector_result, budget, comparison, closure,
            _input_refusal("vector_result_invalid", "A4 result has the wrong record type"))
    if (vector_result.status != "complete" or not vector_result.domains
            or vector_result.refusals
            or vector_result.represented_work != vector_result.planned_work):
        reason = "graph_empty" if not vector_result.domains else "vector_result_invalid"
        detail = ("complete vector result contains no graph edges" if reason == "graph_empty"
                  else "A4 result is not a complete represented domain")
        return _refused_result(
            vector_result, budget, comparison, closure,
            _input_refusal(reason, detail))
    if (vector_result.comparison_tolerance is None
            or float(vector_result.comparison_tolerance) != comparison):
        return _refused_result(
            vector_result, budget, comparison, closure,
            _input_refusal("comparison_tolerance_mismatch",
                            "comparison tolerance differs from the validated A4 request"))

    try:
        replays = plane_vector_domain.validate_lateral_vector_domain_inputs(
            vector_result, support, p1d_result, matches, phase_evidence,
            readings=readings, signatures=signatures,
            comparison_tolerance=comparison)
    except (AttributeError, TypeError, ValueError, OverflowError, FloatingPointError):
        replays = False
    if not replays:
        return _refused_result(
            vector_result, budget, comparison, closure,
            _input_refusal("vector_result_replay_mismatch",
                            "A4 indexed domains do not replay against current sources"))

    domains = vector_result.domains
    identities = tuple(domain.match_identity for domain in domains)
    if len(set(identities)) != len(identities):
        return _refused_result(
            vector_result, budget, comparison, closure,
            _input_refusal("duplicate_match_identity",
                            "A4 contains duplicate directed edge identities"))
    for domain in domains:
        identity = domain.match_identity
        if (type(identity.left_frame) is not str or not identity.left_frame
                or type(identity.right_frame) is not str or not identity.right_frame
                or identity.left_frame == identity.right_frame):
            return _refused_result(
                vector_result, budget, comparison, closure,
                _input_refusal("identity_endpoint_invalid",
                                "edge endpoints must be distinct non-empty frame names",
                                identity))
        try:
            _delta_xyz(domain, 0)
        except _IteratorMismatch as error:
            reason = ("nonfinite_delta" if "non-finite" in str(error)
                      else "factor_contract_invalid")
            return _refused_result(
                vector_result, budget, comparison, closure,
                _input_refusal(reason, "A4 edge factor failed finite axis replay", identity))

    try:
        graph_plan = _make_graph_plan(identities)
    except ValueError:
        return _refused_result(
            vector_result, budget, comparison, closure,
            _input_refusal("factor_contract_invalid",
                            "A4 edge identities do not form a valid canonical graph"))
    domain_by_identity = {domain.match_identity: domain for domain in domains}
    work = _deterministic_work(graph_plan, domain_by_identity)
    if _preflight_budget_exceeds(work, budget):
        return _budget_exceeded_result(
            vector_result, budget, comparison, closure, work,
            C2RBRuntimeWork(0, 0, 0, 0, 0, 0, 0, 0),
            "planned_budget_exceeded")

    runtime = _RuntimeCounter(budget)
    try:
        component_results = []
        closure_fraction = Fraction.from_float(closure)
        for component in graph_plan.components:
            component_ids = (*component.tree_edges, *component.chords)
            component_domains = {identity: domain_by_identity[identity]
                                 for identity in component_ids}
            singleton_classes = {
                identity: _factorized_single_delta_class(domain, runtime)
                for identity, domain in component_domains.items()}
            if all(value is not None for value in singleton_classes.values()):
                result = _singleton_component_result(
                    component, component_domains, singleton_classes,
                    closure_fraction, runtime)
            else:
                result = _solve_general_component(
                    component, component_domains, closure_fraction, runtime)
            component_results.append(result)
        runtime_result = runtime.snapshot()
        if _runtime_over(runtime_result, budget):
            return _budget_exceeded_result(
                vector_result, budget, comparison, closure, work,
                runtime_result, "runtime_count_mismatch")
        return _make_result(
            status="complete", budget=budget, deterministic_work=work,
            runtime_work=runtime_result,
            vector_result_sha256=vector_result.result_sha256,
            vector_input_sha256=vector_result.input_sha256,
            comparison_tolerance=comparison, closure_tolerance=closure,
            components=tuple(component_results))
    except _BudgetStop as error:
        return _budget_exceeded_result(
            vector_result, budget, comparison, closure, work, runtime.snapshot(),
            error.reason)
    except (_IteratorMismatch, AttributeError, IndexError, KeyError, TypeError,
            ValueError, OverflowError, FloatingPointError) as error:
        return _budget_exceeded_result(
            vector_result, budget, comparison, closure, work, runtime.snapshot(),
            "runtime_count_mismatch")


def validate_relative_translation_graph(
    result: C2RBResult,
    vector_result: plane_vector_domain.C2RVectorDomainResult,
    support: plane_axis_support.C1AxisSupport,
    p1d_result: plane_shift_domain.ShiftDomainResult,
    matches: Sequence[plane_graph.PlaneMatch],
    phase_evidence: Sequence[plane_residue.HypothesisPhaseShiftEvidence],
    *,
    readings,
    signatures,
    comparison_tolerance: float,
    closure_tolerance: float,
) -> bool:
    """Replay A4 and the exact bounded solver; compare canonical result bytes."""
    if (not isinstance(result, C2RBResult) or result.status != "complete"
            or result.comparison_tolerance != comparison_tolerance
            or result.closure_tolerance != closure_tolerance):
        return False
    try:
        replay = solve_relative_translation_graph(
            vector_result, support, p1d_result, matches, phase_evidence,
            readings=readings, signatures=signatures,
            comparison_tolerance=comparison_tolerance,
            closure_tolerance=closure_tolerance, budget=result.budget)
        return (replay.status == "complete"
                and replay.to_json_bytes() == result.to_json_bytes())
    except (AttributeError, TypeError, ValueError, OverflowError, FloatingPointError):
        return False
