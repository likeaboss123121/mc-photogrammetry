"""M4 V2-A isolated exact bounded global endpoint-provenance CSP.

This module answers exactly one question: do the accepted C2-R-B edge deltas
admit **zero, one, or multiple** globally compatible endpoint-provenance
assignments over the accepted A4 provenance candidates.  It reports that count
separately from translation uniqueness.

It defines no V2 placement API, no integer or dyadic placement, no phase/seam
anchor, no quantization rule, no signature statistic, no ranking/selection and
no runner/schema/selector/CLI/evaluator/renderer/export integration.

Only index-only A4 factor accessors (``iter_entry_groups``,
``provenance_key_at``, ``vector_at``) and the stored side tables are read.  The
names ``candidate_at``, ``iter_candidates``, ``materialize_lateral_vector_domains``
and ``solve_component`` are denied and must never be called here; no A4 candidate
record is ever materialized.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from mc_photogrammetry.reconstruction import lattice_planes
from mc_photogrammetry.reconstruction import plane_axis_support
from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction import plane_relative_translation
from mc_photogrammetry.reconstruction import plane_residue
from mc_photogrammetry.reconstruction import plane_shift_domain
from mc_photogrammetry.reconstruction import plane_vector_domain


__all__ = [
    "V2ABudget",
    "V2AWork",
    "V2ARefusal",
    "ReadingIdentity",
    "ResidueAxisDomain",
    "EndpointResidueIdentity",
    "EndpointLabel",
    "ReadingLabelDomain",
    "KernelRepresentativeSentinel",
    "KernelBucket",
    "KernelEdge",
    "EndpointAssignmentWitness",
    "EnumerationCommitment",
    "EdgeFeasibleRecord",
    "KernelResult",
    "V2AEndpointResult",
    "solve_endpoint_provenance_kernel",
    "analyze_endpoint_provenance",
    "validate_endpoint_provenance_result",
]


V2A_SCHEMA = "m4-v2a-endpoint-provenance-csp-v1"
DIGEST_LENGTH = 64
MAX_DETAIL_LENGTH = 240
FOOTPRINT_ABSENT = "footprint_unavailable"

STATUSES = frozenset({"complete", "refused", "budget_exhausted"})
OUTCOMES = frozenset({"endpoint_unique", "endpoint_none", "endpoint_multiple"})
EDGE_KINDS = frozenset({"tree", "chord"})
BUDGET_FAILURE_REASONS = frozenset({
    "planned_budget_exceeded", "runtime_budget_exceeded", "runtime_count_mismatch",
})
REFUSAL_REASONS = frozenset({
    "budget_invalid",
    "gate_not_ready",
    "c2rb_result_invalid",
    "c2rb_replay_mismatch",
    "a4_replay_mismatch",
    "canonical_universe_mismatch",
    "identity_replay_mismatch",
    "graph_binding_mismatch",
    "accepted_delta_missing",
    "stored_delta_echo_mismatch",
    "stored_reference_not_feasible",
    "endpoint_source_conflict",
    "reading_axis_missing",
    "signature_replay_mismatch",
    "tree_injectivity_violation",
    "result_validation_mismatch",
    "kernel_input_invalid",
    "input_record_invalid",
})


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
    return (type(value) is str and len(value) == DIGEST_LENGTH
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


def _dyadic_from_float(value: float) -> tuple[int, int]:
    numerator, denominator = float(value).as_integer_ratio()
    divisor = math.gcd(numerator, denominator)
    return numerator // divisor, denominator // divisor


def _validate_dyadic(value: Any) -> bool:
    return (type(value) is tuple and len(value) == 2
            and type(value[0]) is int and type(value[1]) is int
            and value[1] > 0 and math.gcd(value[0], value[1]) == 1)


def _fraction(value: tuple[int, int]) -> Fraction:
    return Fraction(value[0], value[1])


def _dyadic(value: Fraction) -> tuple[int, int]:
    return value.numerator, value.denominator


def _sub_dyadic(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    return _dyadic(_fraction(left) - _fraction(right))


def _dyadics_record(values: Sequence[tuple[int, int]]) -> list[list[int]]:
    return [[numerator, denominator] for numerator, denominator in values]


def _seal(record_type: type, fields: Mapping[str, Any], digest_field: str,
          payload_function) -> Any:
    provisional = object.__new__(record_type)
    for name, value in fields.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, digest_field,
                       _sha256_json(payload_function(provisional)))
    record_type.__post_init__(provisional)
    return provisional


class V2AFailure(Exception):
    """A typed fail-closed refusal inside the isolated V2-A pipeline."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail[:MAX_DETAIL_LENGTH]


class V2ABudgetStop(Exception):
    """An exact runtime budget excess; every partial artifact is discarded."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class V2AKernelInputError(ValueError):
    """The isolated post-gate kernel received malformed typed inputs."""


# ---------------------------------------------------------------------------
# Budgets and work records
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class V2ABudget:
    """Explicit positive whole-request limits.  No limit has a default."""

    max_readings: int
    max_edges: int
    max_endpoint_labels: int
    max_residues_inspected: int
    max_edge_candidates_inspected: int
    max_partial_label_assignments: int
    max_consistency_checks: int
    max_result_validation_cross_links: int
    max_distinct_assignments: int
    max_retained_witnesses: int

    def __post_init__(self) -> None:
        values = (
            self.max_readings, self.max_edges, self.max_endpoint_labels,
            self.max_residues_inspected, self.max_edge_candidates_inspected,
            self.max_partial_label_assignments, self.max_consistency_checks,
            self.max_result_validation_cross_links,
            self.max_distinct_assignments, self.max_retained_witnesses,
        )
        if any(not _strict_int(value, minimum=1) for value in values):
            raise ValueError("V2-A budget limits must be positive exact integers")
        if self.max_distinct_assignments < 2:
            raise ValueError(
                "max_distinct_assignments must be at least two so "
                "endpoint_multiple is certifiable")

    def to_record(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class V2AWork:
    """Machine-checked planned or observed request-wide work."""

    readings: int
    edges: int
    endpoint_labels: int
    residues_inspected: int
    edge_candidates_inspected: int
    partial_label_assignments: int
    consistency_checks: int
    result_validation_cross_links: int
    distinct_assignments: int
    retained_witnesses: int
    candidate_materializations: int

    def __post_init__(self) -> None:
        if any(not _strict_int(value) for value in (
                self.readings, self.edges, self.endpoint_labels,
                self.residues_inspected, self.edge_candidates_inspected,
                self.partial_label_assignments, self.consistency_checks,
                self.result_validation_cross_links, self.distinct_assignments,
                self.retained_witnesses, self.candidate_materializations)):
            raise ValueError("V2-A work counters must be non-negative integers")
        if self.candidate_materializations != 0:
            raise ValueError("V2-A never materializes an A4 candidate record")

    def to_record(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def zero(cls) -> "V2AWork":
        return cls(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


@dataclass(frozen=True, slots=True)
class V2ARefusal:
    """One typed atomic refusal cause."""

    reason: str
    detail: str
    match_identity: plane_axis_support.C1MatchIdentity | None = None

    def __post_init__(self) -> None:
        if (self.reason not in REFUSAL_REASONS or type(self.detail) is not str
                or not self.detail or len(self.detail) > MAX_DETAIL_LENGTH
                or any(ord(character) < 32 or ord(character) == 127
                       for character in self.detail)
                or (self.match_identity is not None
                    and not isinstance(self.match_identity,
                                       plane_axis_support.C1MatchIdentity))):
            raise ValueError("V2-A refusal is malformed")

    def to_record(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "detail": self.detail,
            "match_identity": (None if self.match_identity is None
                               else self.match_identity.to_record()),
        }


def _refusal_key(value: V2ARefusal) -> tuple[Any, ...]:
    identity = value.match_identity
    return (() if identity is None else identity.sort_key(), value.reason,
            value.detail)


# ---------------------------------------------------------------------------
# Canonical typed identity
# ---------------------------------------------------------------------------

def _reading_identity_payload(value: "ReadingIdentity") -> dict[str, Any]:
    return {
        "frame": value.frame,
        "canonical_index": value.canonical_index,
        "reading_order_key_sha256": value.reading_order_key_sha256,
        "reading_axis": value.reading_axis,
        "source_region_id": value.source_region_id,
        "canonical_universe_sha256": value.canonical_universe_sha256,
    }


@dataclass(frozen=True, slots=True)
class ReadingIdentity:
    """Frozen typed reading identity: (frame, index, order key, axis, source)."""

    frame: str
    canonical_index: int
    reading_order_key_sha256: str
    reading_axis: int
    source_region_id: str | None
    canonical_universe_sha256: str
    identity_sha256: str

    def __post_init__(self) -> None:
        if (type(self.frame) is not str or not self.frame
                or not _strict_int(self.canonical_index)
                or not _valid_digest(self.reading_order_key_sha256)
                or type(self.reading_axis) is not int
                or self.reading_axis not in (0, 1, 2)
                or (self.source_region_id is not None
                    and (type(self.source_region_id) is not str
                         or not self.source_region_id))
                or not _valid_digest(self.canonical_universe_sha256)
                or not _valid_digest(self.identity_sha256)
                or _sha256_json(_reading_identity_payload(self))
                != self.identity_sha256):
            raise ValueError("V2-A reading identity is malformed")

    @classmethod
    def build(cls, *, frame: str, canonical_index: int,
              reading_order_key_sha256: str, reading_axis: int,
              source_region_id: str | None,
              canonical_universe_sha256: str) -> "ReadingIdentity":
        return _seal(cls, {
            "frame": frame, "canonical_index": canonical_index,
            "reading_order_key_sha256": reading_order_key_sha256,
            "reading_axis": reading_axis,
            "source_region_id": source_region_id,
            "canonical_universe_sha256": canonical_universe_sha256,
        }, "identity_sha256", _reading_identity_payload)

    def sort_key(self) -> tuple[Any, ...]:
        return (self.frame, self.canonical_index)

    def to_record(self) -> dict[str, Any]:
        record = _reading_identity_payload(self)
        record["identity_sha256"] = self.identity_sha256
        return record


def _residue_axis_payload(value: "ResidueAxisDomain") -> dict[str, Any]:
    return {
        "local_axis": value.local_axis,
        "local_axis_label": value.local_axis_label,
        "selected_b": float(value.selected_b),
        "source_record_sha256": value.source_record_sha256,
        "candidate_values": [float(item) for item in value.candidate_values],
    }


@dataclass(frozen=True, slots=True)
class ResidueAxisDomain:
    """The unique role-neutral P1-B source atom for one (reading, local axis)."""

    local_axis: str
    local_axis_label: int
    selected_b: float
    source_record_sha256: str
    candidate_values: tuple[float, ...]
    domain_sha256: str

    def __post_init__(self) -> None:
        if (self.local_axis not in ("u", "v")
                or type(self.local_axis_label) is not int
                or self.local_axis_label not in (0, 1, 2)
                or _finite(self.selected_b) is None or self.selected_b <= 0.0
                or not _valid_digest(self.source_record_sha256)
                or not isinstance(self.candidate_values, tuple)
                or not self.candidate_values
                or any(_finite(value) is None or not 0.0 <= value < 1.0
                       for value in self.candidate_values)
                or any(left >= right for left, right in zip(
                    self.candidate_values, self.candidate_values[1:]))
                or not _valid_digest(self.domain_sha256)
                or _sha256_json(_residue_axis_payload(self)) != self.domain_sha256):
            raise ValueError("V2-A residue axis domain is malformed")

    @classmethod
    def build(cls, *, local_axis: str, local_axis_label: int, selected_b: float,
              source_record_sha256: str,
              candidate_values: Sequence[float]) -> "ResidueAxisDomain":
        return _seal(cls, {
            "local_axis": local_axis, "local_axis_label": local_axis_label,
            "selected_b": float(selected_b),
            "source_record_sha256": source_record_sha256,
            "candidate_values": tuple(float(item) for item in candidate_values),
        }, "domain_sha256", _residue_axis_payload)

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_values)

    def residue_identity(self, index: int) -> "EndpointResidueIdentity":
        if type(index) is not int or not 0 <= index < self.candidate_count:
            raise IndexError("residue index is outside its axis domain")
        return EndpointResidueIdentity.build(
            source_record_sha256=self.source_record_sha256,
            selected_b=self.selected_b, local_axis=self.local_axis,
            local_axis_label=self.local_axis_label, residue_index=index,
            residue_value=self.candidate_values[index])

    def to_record(self) -> dict[str, Any]:
        record = _residue_axis_payload(self)
        record["domain_sha256"] = self.domain_sha256
        return record


def _endpoint_residue_payload(value: "EndpointResidueIdentity") -> dict[str, Any]:
    return {
        "source_record_sha256": value.source_record_sha256,
        "selected_b": float(value.selected_b),
        "local_axis": value.local_axis,
        "local_axis_label": value.local_axis_label,
        "residue_index": value.residue_index,
        "residue_value": float(value.residue_value),
    }


@dataclass(frozen=True, slots=True)
class EndpointResidueIdentity:
    """Role-neutral endpoint residue identity (no endpoint_role, no table SHA)."""

    source_record_sha256: str
    selected_b: float
    local_axis: str
    local_axis_label: int
    residue_index: int
    residue_value: float
    residue_sha256: str

    def __post_init__(self) -> None:
        if (not _valid_digest(self.source_record_sha256)
                or _finite(self.selected_b) is None or self.selected_b <= 0.0
                or self.local_axis not in ("u", "v")
                or type(self.local_axis_label) is not int
                or self.local_axis_label not in (0, 1, 2)
                or not _strict_int(self.residue_index)
                or _finite(self.residue_value) is None
                or not 0.0 <= self.residue_value < 1.0
                or not _valid_digest(self.residue_sha256)
                or _sha256_json(_endpoint_residue_payload(self))
                != self.residue_sha256):
            raise ValueError("V2-A endpoint residue identity is malformed")

    @classmethod
    def build(cls, *, source_record_sha256: str, selected_b: float,
              local_axis: str, local_axis_label: int, residue_index: int,
              residue_value: float) -> "EndpointResidueIdentity":
        return _seal(cls, {
            "source_record_sha256": source_record_sha256,
            "selected_b": float(selected_b), "local_axis": local_axis,
            "local_axis_label": local_axis_label,
            "residue_index": residue_index,
            "residue_value": float(residue_value),
        }, "residue_sha256", _endpoint_residue_payload)

    def to_record(self) -> dict[str, Any]:
        record = _endpoint_residue_payload(self)
        record["residue_sha256"] = self.residue_sha256
        return record


def _endpoint_label_payload(value: "EndpointLabel") -> dict[str, Any]:
    return {
        "reading_identity": value.reading_identity.to_record(),
        "selected_b": float(value.selected_b),
        "absolute_quadrant": value.absolute_quadrant,
        "residue_u": value.residue_u.to_record(),
        "residue_v": value.residue_v.to_record(),
        "footprint_identity": value.footprint_identity,
    }


@dataclass(frozen=True, slots=True)
class EndpointLabel:
    """One complete endpoint label; the free part is (residue_u, residue_v)."""

    reading_identity: ReadingIdentity
    selected_b: float
    absolute_quadrant: int
    residue_u: EndpointResidueIdentity
    residue_v: EndpointResidueIdentity
    footprint_identity: str
    label_sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(self.reading_identity, ReadingIdentity)
                or _finite(self.selected_b) is None or self.selected_b <= 0.0
                or type(self.absolute_quadrant) is not int
                or self.absolute_quadrant not in (0, 1, 2, 3)
                or not isinstance(self.residue_u, EndpointResidueIdentity)
                or not isinstance(self.residue_v, EndpointResidueIdentity)
                or self.residue_u.local_axis != "u"
                or self.residue_v.local_axis != "v"
                or self.residue_u.selected_b != self.selected_b
                or self.residue_v.selected_b != self.selected_b
                or (self.footprint_identity != FOOTPRINT_ABSENT
                    and not _valid_digest(self.footprint_identity))
                or not _valid_digest(self.label_sha256)
                or _sha256_json(_endpoint_label_payload(self)) != self.label_sha256):
            raise ValueError("V2-A endpoint label is malformed")

    @classmethod
    def build(cls, *, reading_identity: ReadingIdentity, selected_b: float,
              absolute_quadrant: int, residue_u: EndpointResidueIdentity,
              residue_v: EndpointResidueIdentity,
              footprint_identity: str) -> "EndpointLabel":
        return _seal(cls, {
            "reading_identity": reading_identity,
            "selected_b": float(selected_b),
            "absolute_quadrant": absolute_quadrant,
            "residue_u": residue_u, "residue_v": residue_v,
            "footprint_identity": footprint_identity,
        }, "label_sha256", _endpoint_label_payload)

    def to_record(self) -> dict[str, Any]:
        record = _endpoint_label_payload(self)
        record["label_sha256"] = self.label_sha256
        return record


def _reading_domain_payload(value: "ReadingLabelDomain") -> dict[str, Any]:
    return {
        "reading_identity": value.reading_identity.to_record(),
        "selected_b": float(value.selected_b),
        "absolute_quadrant": value.absolute_quadrant,
        "footprint_identity": value.footprint_identity,
        "residue_u": value.residue_u.to_record(),
        "residue_v": value.residue_v.to_record(),
        "label_count": value.label_count,
    }


@dataclass(frozen=True, slots=True)
class ReadingLabelDomain:
    """One reading's finite label domain ``{pinned} x {(i_u, i_v)}``."""

    reading_identity: ReadingIdentity
    selected_b: float
    absolute_quadrant: int
    footprint_identity: str
    residue_u: ResidueAxisDomain
    residue_v: ResidueAxisDomain
    label_count: int
    domain_sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(self.reading_identity, ReadingIdentity)
                or _finite(self.selected_b) is None or self.selected_b <= 0.0
                or type(self.absolute_quadrant) is not int
                or self.absolute_quadrant not in (0, 1, 2, 3)
                or (self.footprint_identity != FOOTPRINT_ABSENT
                    and not _valid_digest(self.footprint_identity))
                or not isinstance(self.residue_u, ResidueAxisDomain)
                or not isinstance(self.residue_v, ResidueAxisDomain)
                or self.residue_u.local_axis != "u"
                or self.residue_v.local_axis != "v"
                or self.residue_u.selected_b != self.selected_b
                or self.residue_v.selected_b != self.selected_b
                or type(self.label_count) is not int
                or self.label_count != (self.residue_u.candidate_count
                                        * self.residue_v.candidate_count)
                or not _valid_digest(self.domain_sha256)
                or _sha256_json(_reading_domain_payload(self)) != self.domain_sha256):
            raise ValueError("V2-A reading label domain is malformed")

    @classmethod
    def build(cls, *, reading_identity: ReadingIdentity, selected_b: float,
              absolute_quadrant: int, footprint_identity: str,
              residue_u: ResidueAxisDomain,
              residue_v: ResidueAxisDomain) -> "ReadingLabelDomain":
        return _seal(cls, {
            "reading_identity": reading_identity,
            "selected_b": float(selected_b),
            "absolute_quadrant": absolute_quadrant,
            "footprint_identity": footprint_identity,
            "residue_u": residue_u, "residue_v": residue_v,
            "label_count": (residue_u.candidate_count
                            * residue_v.candidate_count),
        }, "domain_sha256", _reading_domain_payload)

    @property
    def reading_sha256(self) -> str:
        return self.reading_identity.identity_sha256

    @property
    def code_count(self) -> int:
        return self.label_count

    def code_for(self, index_u: int, index_v: int) -> int:
        if (type(index_u) is not int or type(index_v) is not int
                or not 0 <= index_u < self.residue_u.candidate_count
                or not 0 <= index_v < self.residue_v.candidate_count):
            raise IndexError("free residue indices are outside the reading domain")
        return index_u * self.residue_v.candidate_count + index_v

    def decode_code(self, code: int) -> tuple[int, int]:
        if type(code) is not int or not 0 <= code < self.label_count:
            raise IndexError("free index code is outside the reading domain")
        return divmod(code, self.residue_v.candidate_count)

    def iter_codes(self):
        for index_u in range(self.residue_u.candidate_count):
            base = index_u * self.residue_v.candidate_count
            for index_v in range(self.residue_v.candidate_count):
                yield base + index_v

    def label_for(self, index_u: int, index_v: int) -> EndpointLabel:
        return EndpointLabel.build(
            reading_identity=self.reading_identity,
            selected_b=self.selected_b,
            absolute_quadrant=self.absolute_quadrant,
            residue_u=self.residue_u.residue_identity(index_u),
            residue_v=self.residue_v.residue_identity(index_v),
            footprint_identity=self.footprint_identity)

    def label_for_code(self, code: int) -> EndpointLabel:
        index_u, index_v = self.decode_code(code)
        return self.label_for(index_u, index_v)

    def to_record(self) -> dict[str, Any]:
        record = _reading_domain_payload(self)
        record["domain_sha256"] = self.domain_sha256
        return record


# ---------------------------------------------------------------------------
# Isolated post-gate kernel records
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class KernelRepresentativeSentinel:
    """Opaque C2-R-B stored-reference echo.  Never evidence in the kernel."""

    global_index: int | None
    residue_indices: tuple[int, int, int, int] | None
    provenance_sha256: str | None

    def __post_init__(self) -> None:
        if (self.global_index is not None
                and not _strict_int(self.global_index)):
            raise ValueError("representative sentinel index is invalid")
        if (self.residue_indices is not None
                and (type(self.residue_indices) is not tuple
                     or len(self.residue_indices) != 4
                     or any(not _strict_int(value)
                            for value in self.residue_indices))):
            raise ValueError("representative sentinel residues are invalid")
        if (self.provenance_sha256 is not None
                and not _valid_digest(self.provenance_sha256)):
            raise ValueError("representative sentinel provenance is invalid")


def _kernel_bucket_payload(value: "KernelBucket") -> dict[str, Any]:
    return {
        "global_index": value.global_index,
        "left_code": value.left_code,
        "right_code": value.right_code,
        "multiplicity": value.multiplicity,
    }


@dataclass(frozen=True, slots=True)
class KernelBucket:
    """One induced endpoint-label pair with its exact witness multiplicity."""

    global_index: int
    left_code: int
    right_code: int
    multiplicity: int
    bucket_sha256: str

    def __post_init__(self) -> None:
        if (not _strict_int(self.global_index)
                or not _strict_int(self.left_code)
                or not _strict_int(self.right_code)
                or not _strict_int(self.multiplicity, minimum=1)
                or not _valid_digest(self.bucket_sha256)
                or _sha256_json(_kernel_bucket_payload(self)) != self.bucket_sha256):
            raise ValueError("V2-A kernel bucket is malformed")

    @classmethod
    def build(cls, *, global_index: int, left_code: int, right_code: int,
              multiplicity: int) -> "KernelBucket":
        return _seal(cls, {
            "global_index": global_index, "left_code": left_code,
            "right_code": right_code, "multiplicity": multiplicity,
        }, "bucket_sha256", _kernel_bucket_payload)

    def to_record(self) -> dict[str, Any]:
        record = _kernel_bucket_payload(self)
        record["bucket_sha256"] = self.bucket_sha256
        return record


def _kernel_edge_payload(value: "KernelEdge") -> dict[str, Any]:
    return {
        "match_identity": value.match_identity.to_record(),
        "edge_kind": value.edge_kind,
        "accepted_delta_dyadic_xyz": _dyadics_record(
            value.accepted_delta_dyadic_xyz),
        "left_reading_sha256": value.left_reading_sha256,
        "right_reading_sha256": value.right_reading_sha256,
        "buckets": [bucket.to_record() for bucket in value.buckets],
    }


@dataclass(frozen=True, slots=True)
class KernelEdge:
    """One edge's accepted delta and complete feasible label-pair buckets.

    ``ignored_representative`` is retained only so the isolated kernel
    independence oracle can vary an opaque sentinel; the kernel never reads it
    and it is excluded from this record's digest.
    """

    match_identity: plane_axis_support.C1MatchIdentity
    edge_kind: str
    accepted_delta_dyadic_xyz: tuple[
        tuple[int, int], tuple[int, int], tuple[int, int]]
    left_reading_sha256: str
    right_reading_sha256: str
    buckets: tuple[KernelBucket, ...]
    edge_sha256: str
    ignored_representative: object | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.match_identity,
                           plane_axis_support.C1MatchIdentity)
                or self.edge_kind not in EDGE_KINDS
                or type(self.accepted_delta_dyadic_xyz) is not tuple
                or len(self.accepted_delta_dyadic_xyz) != 3
                or any(not _validate_dyadic(value)
                       for value in self.accepted_delta_dyadic_xyz)
                or not _valid_digest(self.left_reading_sha256)
                or not _valid_digest(self.right_reading_sha256)
                or not isinstance(self.buckets, tuple)
                or any(not isinstance(bucket, KernelBucket)
                       for bucket in self.buckets)
                or not _valid_digest(self.edge_sha256)
                or _sha256_json(_kernel_edge_payload(self)) != self.edge_sha256):
            raise ValueError("V2-A kernel edge is malformed")

    @classmethod
    def build(cls, *, match_identity: plane_axis_support.C1MatchIdentity,
              edge_kind: str,
              accepted_delta_dyadic_xyz: Sequence[tuple[int, int]],
              left_reading_sha256: str, right_reading_sha256: str,
              buckets: Sequence[KernelBucket],
              ignored_representative: object | None = None) -> "KernelEdge":
        provisional = object.__new__(cls)
        fields = {
            "match_identity": match_identity, "edge_kind": edge_kind,
            "accepted_delta_dyadic_xyz": tuple(
                tuple(item) for item in accepted_delta_dyadic_xyz),
            "left_reading_sha256": left_reading_sha256,
            "right_reading_sha256": right_reading_sha256,
            "buckets": tuple(buckets),
            "ignored_representative": ignored_representative,
        }
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "edge_sha256",
                           _sha256_json(_kernel_edge_payload(provisional)))
        cls.__post_init__(provisional)
        return provisional

    def to_record(self) -> dict[str, Any]:
        record = _kernel_edge_payload(self)
        record["edge_sha256"] = self.edge_sha256
        return record


def _witness_payload(value: "EndpointAssignmentWitness") -> dict[str, Any]:
    return {
        "labels": [[reading_sha256, index_u, index_v, label_sha256]
                   for reading_sha256, index_u, index_v, label_sha256
                   in value.labels],
        "edge_multiplicities": [
            [edge_sha256, multiplicity]
            for edge_sha256, multiplicity in value.edge_multiplicities],
    }


@dataclass(frozen=True, slots=True)
class EndpointAssignmentWitness:
    """One compatible label assignment plus exact per-edge witness multiplicity."""

    labels: tuple[tuple[str, int, int, str], ...]
    edge_multiplicities: tuple[tuple[str, int], ...]
    witness_sha256: str

    def __post_init__(self) -> None:
        if (type(self.labels) is not tuple
                or any(type(item) is not tuple or len(item) != 4
                       or not _valid_digest(item[0])
                       or not _strict_int(item[1]) or not _strict_int(item[2])
                       or not _valid_digest(item[3])
                       for item in self.labels)
                or type(self.edge_multiplicities) is not tuple
                or any(type(item) is not tuple or len(item) != 2
                       or not _valid_digest(item[0])
                       or not _strict_int(item[1], minimum=1)
                       for item in self.edge_multiplicities)
                or not _valid_digest(self.witness_sha256)
                or _sha256_json(_witness_payload(self)) != self.witness_sha256):
            raise ValueError("V2-A endpoint assignment witness is malformed")

    @classmethod
    def build(cls, *, labels: Sequence[tuple[str, int, int, str]],
              edge_multiplicities: Sequence[tuple[str, int]],
              ) -> "EndpointAssignmentWitness":
        return _seal(cls, {
            "labels": tuple(tuple(item) for item in labels),
            "edge_multiplicities": tuple(
                (item[0], item[1]) for item in edge_multiplicities),
        }, "witness_sha256", _witness_payload)

    def to_record(self) -> dict[str, Any]:
        record = _witness_payload(self)
        record["witness_sha256"] = self.witness_sha256
        return record


def _kernel_result_payload(value: "KernelResult") -> dict[str, Any]:
    return {
        "schema": V2A_SCHEMA,
        "status": value.status,
        "classification": value.classification,
        "exhaustive": value.exhaustive,
        "distinct_assignment_count": value.distinct_assignment_count,
        "lower_bound_only": value.lower_bound_only,
        "witnesses": [witness.to_record() for witness in value.witnesses],
        "runtime_work": value.runtime_work.to_record(),
        "budget_failure_reason": value.budget_failure_reason,
        "refusal_reason": value.refusal_reason,
    }


@dataclass(frozen=True, slots=True)
class KernelResult:
    """The isolated post-gate CSP kernel outcome."""

    status: str
    classification: str | None
    exhaustive: bool
    distinct_assignment_count: int
    lower_bound_only: bool
    witnesses: tuple[EndpointAssignmentWitness, ...]
    runtime_work: V2AWork
    budget_failure_reason: str | None
    refusal_reason: str | None
    result_sha256: str

    def __post_init__(self) -> None:
        if (self.status not in STATUSES
                or type(self.exhaustive) is not bool
                or not _strict_int(self.distinct_assignment_count)
                or type(self.lower_bound_only) is not bool
                or type(self.witnesses) is not tuple
                or any(not isinstance(value, EndpointAssignmentWitness)
                       for value in self.witnesses)
                or not isinstance(self.runtime_work, V2AWork)
                or not _valid_digest(self.result_sha256)
                or _sha256_json(_kernel_result_payload(self)) != self.result_sha256):
            raise ValueError("V2-A kernel result is malformed")
        if any(_sha256_json(_witness_payload(witness)) != witness.witness_sha256
               for witness in self.witnesses):
            raise ValueError("V2-A kernel result carries a tampered witness")
        if self.status == "complete":
            if (self.classification not in OUTCOMES
                    or self.budget_failure_reason is not None
                    or self.refusal_reason is not None):
                raise ValueError("complete V2-A kernel result is not a decision")
            if self.classification == "endpoint_unique":
                valid = (self.exhaustive and not self.lower_bound_only
                         and self.distinct_assignment_count == 1
                         and len(self.witnesses) == 1)
            elif self.classification == "endpoint_none":
                valid = (self.exhaustive and not self.lower_bound_only
                         and self.distinct_assignment_count == 0
                         and not self.witnesses)
            else:
                valid = (not self.exhaustive and self.lower_bound_only
                         and self.distinct_assignment_count >= 2
                         and len(self.witnesses) == self.distinct_assignment_count)
            if not valid:
                raise ValueError(
                    "V2-A kernel classification disagrees with its proof records")
        elif self.status == "refused":
            if (self.classification is not None or self.witnesses
                    or self.budget_failure_reason is not None
                    or self.refusal_reason not in REFUSAL_REASONS):
                raise ValueError("refused V2-A kernel result must be atomic")
        else:
            if (self.classification is not None or self.witnesses
                    or self.refusal_reason is not None
                    or self.budget_failure_reason not in BUDGET_FAILURE_REASONS):
                raise ValueError("budget-exhausted V2-A kernel result must be atomic")

    @classmethod
    def build(cls, *, status: str, classification: str | None,
              exhaustive: bool, distinct_assignment_count: int,
              lower_bound_only: bool,
              witnesses: Sequence[EndpointAssignmentWitness],
              runtime_work: V2AWork, budget_failure_reason: str | None,
              refusal_reason: str | None) -> "KernelResult":
        return _seal(cls, {
            "status": status, "classification": classification,
            "exhaustive": exhaustive,
            "distinct_assignment_count": distinct_assignment_count,
            "lower_bound_only": lower_bound_only,
            "witnesses": tuple(witnesses),
            "runtime_work": runtime_work,
            "budget_failure_reason": budget_failure_reason,
            "refusal_reason": refusal_reason,
        }, "result_sha256", _kernel_result_payload)

    def to_record(self) -> dict[str, Any]:
        record = _kernel_result_payload(self)
        record["result_sha256"] = self.result_sha256
        return record

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.to_record(), sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Runtime counters
# ---------------------------------------------------------------------------

class _RuntimeCounter:
    __slots__ = (
        "budget", "readings", "edges", "endpoint_labels", "residues_inspected",
        "edge_candidates_inspected", "partial_label_assignments",
        "consistency_checks", "result_validation_cross_links",
        "distinct_assignments", "retained_witnesses",
    )

    def __init__(self, budget: V2ABudget):
        self.budget = budget
        self.readings = 0
        self.edges = 0
        self.endpoint_labels = 0
        self.residues_inspected = 0
        self.edge_candidates_inspected = 0
        self.partial_label_assignments = 0
        self.consistency_checks = 0
        self.result_validation_cross_links = 0
        self.distinct_assignments = 0
        self.retained_witnesses = 0

    def seed(self, work: V2AWork | None) -> None:
        if work is None:
            return
        self.residues_inspected = work.residues_inspected
        self.edge_candidates_inspected = work.edge_candidates_inspected

    def snapshot(self) -> V2AWork:
        return V2AWork(
            self.readings, self.edges, self.endpoint_labels,
            self.residues_inspected, self.edge_candidates_inspected,
            self.partial_label_assignments, self.consistency_checks,
            self.result_validation_cross_links, self.distinct_assignments,
            self.retained_witnesses, 0)

    def exceeds(self) -> bool:
        limits = self.budget
        return (
            self.readings > limits.max_readings
            or self.edges > limits.max_edges
            or self.endpoint_labels > limits.max_endpoint_labels
            or self.residues_inspected > limits.max_residues_inspected
            or self.edge_candidates_inspected
                > limits.max_edge_candidates_inspected
            or self.partial_label_assignments
                > limits.max_partial_label_assignments
            or self.consistency_checks > limits.max_consistency_checks
            or self.result_validation_cross_links
                > limits.max_result_validation_cross_links
            or self.distinct_assignments > limits.max_distinct_assignments
            or self.retained_witnesses > limits.max_retained_witnesses
        )

    def check(self) -> None:
        if self.exceeds():
            raise V2ABudgetStop("runtime_budget_exceeded")


# ---------------------------------------------------------------------------
# Isolated post-gate CSP kernel
# ---------------------------------------------------------------------------

def _canonical_domains(
    domains: Sequence[ReadingLabelDomain],
) -> tuple[ReadingLabelDomain, ...]:
    if not isinstance(domains, Sequence):
        raise V2AKernelInputError("reading domains must be a sequence")
    if any(not isinstance(domain, ReadingLabelDomain) for domain in domains):
        raise V2AKernelInputError("reading domains contain a foreign record")
    keys = [domain.reading_identity.sort_key() for domain in domains]
    if len(set(keys)) != len(keys):
        raise V2AKernelInputError("reading domains repeat one reading identity")
    return tuple(sorted(domains,
                        key=lambda domain: domain.reading_identity.sort_key()))


def _canonical_edges(edges: Sequence[KernelEdge]) -> tuple[KernelEdge, ...]:
    if not isinstance(edges, Sequence):
        raise V2AKernelInputError("edges must be a sequence")
    if any(not isinstance(edge, KernelEdge) for edge in edges):
        raise V2AKernelInputError("edges contain a foreign record")
    keys = [edge.match_identity.sort_key() for edge in edges]
    if len(set(keys)) != len(keys):
        raise V2AKernelInputError("edges repeat one C1 match identity")
    return tuple(sorted(edges, key=lambda edge: edge.match_identity.sort_key()))


def _kernel_cross_link_mismatch(
    domains: Sequence[ReadingLabelDomain],
    edges: Sequence[KernelEdge],
    witnesses: Sequence[EndpointAssignmentWitness],
) -> str | None:
    """Frozen 4.5 cross-link validation at the isolated kernel boundary."""
    domain_by_sha: dict[str, ReadingLabelDomain] = {}
    for domain in domains:
        if domain.reading_sha256 in domain_by_sha:
            return "kernel reading domains are not injective"
        domain_by_sha[domain.reading_sha256] = domain
    edge_by_sha: dict[str, KernelEdge] = {}
    for edge in edges:
        if edge.edge_sha256 in edge_by_sha:
            return "kernel edges are not injective"
        edge_by_sha[edge.edge_sha256] = edge
    for witness in witnesses:
        if len(witness.labels) != len(domains):
            return "a kernel witness does not label every reading"
        codes: dict[str, int] = {}
        for reading_sha256, index_u, index_v, label_sha256 in witness.labels:
            domain = domain_by_sha.get(reading_sha256)
            if domain is None:
                return "a kernel witness labels an absent reading"
            try:
                expected = domain.label_for(index_u, index_v)
            except (IndexError, ValueError):
                return "a kernel witness label is outside its reading domain"
            if expected.label_sha256 != label_sha256:
                return "a kernel witness label digest disagrees with its domain"
            codes[reading_sha256] = domain.code_for(index_u, index_v)
        if len(witness.edge_multiplicities) != len(edges):
            return "a kernel witness does not cover every edge"
        for edge_sha256, multiplicity in witness.edge_multiplicities:
            edge = edge_by_sha.get(edge_sha256)
            if edge is None:
                return "a kernel witness references an absent edge"
            pair = next(
                (bucket for bucket in edge.buckets
                 if bucket.left_code == codes.get(edge.left_reading_sha256)
                 and bucket.right_code == codes.get(edge.right_reading_sha256)),
                None)
            if pair is None or pair.multiplicity != multiplicity:
                return "a kernel witness multiplicity disagrees with its edge"
    return None


def solve_endpoint_provenance_kernel(
    *,
    domains: Sequence[ReadingLabelDomain],
    edges: Sequence[KernelEdge],
    budget: V2ABudget,
    initial_work: V2AWork | None = None,
    ignored_representative_sentinel: object | None = None,
) -> KernelResult:
    """Solve the induced reading-label CSP exactly, lazily and budgeted.

    The ``ignored_representative_sentinel`` argument and every
    ``KernelEdge.ignored_representative`` field are deliberately never read.
    """
    if not isinstance(budget, V2ABudget):
        raise ValueError("an explicit valid V2ABudget is required")
    counter = _RuntimeCounter(budget)
    try:
        ordered_domains = _canonical_domains(domains)
        ordered_edges = _canonical_edges(edges)
        counter.seed(initial_work)
        counter.readings = len(ordered_domains)
        counter.edges = len(ordered_edges)
        counter.endpoint_labels = sum(domain.label_count
                                      for domain in ordered_domains)
        index_by_sha = {domain.reading_sha256: index
                        for index, domain in enumerate(ordered_domains)}
        if len(index_by_sha) != len(ordered_domains):
            raise V2AKernelInputError("reading identities are not injective")
        codes = tuple(tuple(domain.iter_codes()) for domain in ordered_domains)
        incident: list[list[tuple[int, int]]] = [
            [] for _ in range(len(ordered_domains))]
        edge_multiplicity: list[dict[tuple[int, int], int]] = []
        edge_order: list[tuple[int, int]] = []
        for edge in ordered_edges:
            left = index_by_sha.get(edge.left_reading_sha256)
            right = index_by_sha.get(edge.right_reading_sha256)
            if left is None or right is None or left == right:
                raise V2AKernelInputError(
                    "kernel edge endpoints are missing or self-referential")
            multiplicity: dict[tuple[int, int], int] = {}
            for bucket in edge.buckets:
                if (not 0 <= bucket.left_code < len(codes[left])
                        or not 0 <= bucket.right_code < len(codes[right])):
                    raise V2AKernelInputError(
                        "kernel bucket code is outside its reading domain")
                key = (bucket.left_code, bucket.right_code)
                if edge.edge_kind == "tree" and key in multiplicity:
                    raise V2AKernelInputError(
                        "tree edge has two witnesses for one label pair")
                multiplicity[key] = multiplicity.get(key, 0) + bucket.multiplicity
            edge_multiplicity.append(multiplicity)
            edge_order.append((left, right))
            incident[left].append((len(edge_multiplicity) - 1, 0))
            incident[right].append((len(edge_multiplicity) - 1, 1))
        left_allow: list[dict[int, frozenset[int]]] = []
        right_allow: list[dict[int, frozenset[int]]] = []
        for multiplicity in edge_multiplicity:
            forward: dict[int, set[int]] = {}
            backward: dict[int, set[int]] = {}
            for (left_code, right_code) in multiplicity:
                forward.setdefault(left_code, set()).add(right_code)
                backward.setdefault(right_code, set()).add(left_code)
            left_allow.append({key: frozenset(value)
                               for key, value in forward.items()})
            right_allow.append({key: frozenset(value)
                                for key, value in backward.items()})
        counter.check()
    except V2ABudgetStop as stop:
        return KernelResult.build(
            status="budget_exhausted", classification=None, exhaustive=False,
            distinct_assignment_count=0, lower_bound_only=False, witnesses=(),
            runtime_work=counter.snapshot(), budget_failure_reason=stop.reason,
            refusal_reason=None)
    except V2AKernelInputError:
        return KernelResult.build(
            status="refused", classification=None, exhaustive=False,
            distinct_assignment_count=0, lower_bound_only=False, witnesses=(),
            runtime_work=V2AWork.zero(), budget_failure_reason=None,
            refusal_reason="kernel_input_invalid")

    reading_count = len(ordered_domains)
    assignment = [-1] * reading_count
    witnesses: list[EndpointAssignmentWitness] = []
    found = 0
    stop_reason: str | None = None

    def _consistent(index: int) -> bool:
        for edge_index, role in incident[index]:
            left_index, right_index = edge_order[edge_index]
            multiplicity = edge_multiplicity[edge_index]
            counter.consistency_checks += 1
            counter.check()
            if role == 0:
                code = assignment[index]
                other = assignment[right_index]
                if other < 0:
                    if not left_allow[edge_index].get(code):
                        return False
                elif (code, other) not in multiplicity:
                    return False
            else:
                code = assignment[index]
                other = assignment[left_index]
                if other < 0:
                    if not right_allow[edge_index].get(code):
                        return False
                elif (other, code) not in multiplicity:
                    return False
        return True

    def _record_assignment() -> EndpointAssignmentWitness:
        labels = []
        for index, domain in enumerate(ordered_domains):
            index_u, index_v = domain.decode_code(assignment[index])
            labels.append((domain.reading_sha256, index_u, index_v,
                           domain.label_for(index_u, index_v).label_sha256))
        edge_multiplicities = []
        for edge_index, edge in enumerate(ordered_edges):
            key = (assignment[edge_order[edge_index][0]],
                   assignment[edge_order[edge_index][1]])
            edge_multiplicities.append(
                (edge.edge_sha256, edge_multiplicity[edge_index][key]))
        return EndpointAssignmentWitness.build(
            labels=labels, edge_multiplicities=edge_multiplicities)

    try:
        if reading_count == 0:
            return KernelResult.build(
                status="complete", classification="endpoint_none",
                exhaustive=True, distinct_assignment_count=0,
                lower_bound_only=False, witnesses=(),
                runtime_work=counter.snapshot(), budget_failure_reason=None,
                refusal_reason=None)
        index = 0
        iterators = [iter(codes[0])]
        while True:
            if index == reading_count:
                found += 1
                counter.distinct_assignments = found
                witnesses.append(_record_assignment())
                counter.retained_witnesses = len(witnesses)
                counter.check()
                if found >= 2:
                    break
                index -= 1
                assignment[index] = -1
                continue
            progressed = False
            for code in iterators[index]:
                assignment[index] = code
                counter.partial_label_assignments += 1
                counter.check()
                if _consistent(index):
                    progressed = True
                    break
                assignment[index] = -1
            if progressed:
                index += 1
                if index < reading_count:
                    iterators.append(iter(codes[index]))
            else:
                assignment[index] = -1
                iterators.pop()
                index -= 1
                if index < 0:
                    break
                assignment[index] = -1
    except V2ABudgetStop as stop:
        stop_reason = stop.reason

    if stop_reason is not None:
        return KernelResult.build(
            status="budget_exhausted", classification=None, exhaustive=False,
            distinct_assignment_count=0, lower_bound_only=False, witnesses=(),
            runtime_work=counter.snapshot(), budget_failure_reason=stop_reason,
            refusal_reason=None)
    final_work = counter.snapshot()
    if (final_work.retained_witnesses != len(witnesses)
            or final_work.residues_inspected
            != 4 * final_work.edge_candidates_inspected):
        return KernelResult.build(
            status="budget_exhausted", classification=None, exhaustive=False,
            distinct_assignment_count=0, lower_bound_only=False, witnesses=(),
            runtime_work=final_work,
            budget_failure_reason="runtime_count_mismatch", refusal_reason=None)
    mismatch = _kernel_cross_link_mismatch(
        ordered_domains, ordered_edges, tuple(witnesses))
    if mismatch is not None:
        return KernelResult.build(
            status="refused", classification=None, exhaustive=False,
            distinct_assignment_count=0, lower_bound_only=False, witnesses=(),
            runtime_work=final_work, budget_failure_reason=None,
            refusal_reason="result_validation_mismatch")
    if found == 0:
        return KernelResult.build(
            status="complete", classification="endpoint_none", exhaustive=True,
            distinct_assignment_count=0, lower_bound_only=False, witnesses=(),
            runtime_work=final_work, budget_failure_reason=None,
            refusal_reason=None)
    if found == 1:
        return KernelResult.build(
            status="complete", classification="endpoint_unique", exhaustive=True,
            distinct_assignment_count=1, lower_bound_only=False,
            witnesses=tuple(witnesses), runtime_work=final_work,
            budget_failure_reason=None, refusal_reason=None)
    return KernelResult.build(
        status="complete", classification="endpoint_multiple", exhaustive=False,
        distinct_assignment_count=found, lower_bound_only=True,
        witnesses=tuple(witnesses), runtime_work=final_work,
        budget_failure_reason=None, refusal_reason=None)


# ---------------------------------------------------------------------------
# Public replay pipeline
# ---------------------------------------------------------------------------

def _canonical_universe_sha256(
    readings: Sequence[lattice_planes.PlaneReading],
) -> str:
    if not isinstance(readings, Sequence) or not readings:
        raise V2AFailure("canonical_universe_mismatch",
                         "complete reading universe is missing")
    by_frame: dict[str, list[lattice_planes.PlaneReading]] = {}
    for reading in readings:
        if not isinstance(reading, lattice_planes.PlaneReading):
            raise V2AFailure("canonical_universe_mismatch",
                             "complete reading universe contains a foreign object")
        if type(reading.frame) is not str or not reading.frame:
            raise V2AFailure("canonical_universe_mismatch",
                             "reading frame identity is invalid")
        by_frame.setdefault(reading.frame, []).append(reading)
    payload = []
    for frame in sorted(by_frame):
        try:
            ordered = tuple(sorted(by_frame[frame],
                                   key=lattice_planes.reading_order_key))
            keys = tuple(lattice_planes.reading_order_key(item)
                         for item in ordered)
            if len(set(keys)) != len(keys):
                raise ValueError("ambiguous reading order keys")
            for key in keys:
                _json_value(key)
        except (TypeError, ValueError, OverflowError) as error:
            raise V2AFailure("canonical_universe_mismatch",
                             "complete reading universe cannot be canonicalized") \
                from error
        payload.extend({"frame": frame, "index": index, "reading_order_key": key}
                       for index, key in enumerate(keys))
    return _sha256_json(payload)


def _ordered_readings(
    readings: Sequence[lattice_planes.PlaneReading],
) -> dict[str, tuple[lattice_planes.PlaneReading, ...]]:
    by_frame: dict[str, list[lattice_planes.PlaneReading]] = {}
    for reading in readings:
        by_frame.setdefault(reading.frame, []).append(reading)
    return {frame: tuple(sorted(values, key=lattice_planes.reading_order_key))
            for frame, values in by_frame.items()}


def _reading_order_key_sha256(reading: lattice_planes.PlaneReading) -> str:
    return _sha256_json(lattice_planes.reading_order_key(reading))


def _canonical_index_lookup(
    ordered: Mapping[str, tuple[lattice_planes.PlaneReading, ...]],
) -> dict[tuple[str, tuple[Any, ...]], int]:
    """Map each reading's content key to its canonical per-frame index.

    The key is the frozen typed precursor ``(frame, reading_order_key)``.  No
    Python object identity is consulted; two independently constructed readings
    with equal content share one canonical index, exactly as
    ``_canonical_universe_sha256`` orders them.
    """
    lookup: dict[tuple[str, tuple[Any, ...]], int] = {}
    for frame, values in ordered.items():
        for position, candidate in enumerate(values):
            key = (frame, lattice_planes.reading_order_key(candidate))
            if key in lookup:
                raise V2AFailure("canonical_universe_mismatch",
                                 "ambiguous reading order keys")
            lookup[key] = position
    return lookup


def _canonical_position(
    reading: lattice_planes.PlaneReading,
    lookup: Mapping[tuple[str, tuple[Any, ...]], int],
) -> int:
    key = (reading.frame, lattice_planes.reading_order_key(reading))
    position = lookup.get(key)
    if position is None:
        raise V2AFailure("identity_replay_mismatch",
                         "endpoint reading is not in the canonical universe")
    return position


def _footprint_identity(
    reading: lattice_planes.PlaneReading, selected_b: float,
    signatures: Sequence[lattice_planes.PlaneSignature],
) -> str:
    reading_key = (reading.frame, _reading_order_key_sha256(reading))
    matches = []
    for signature in signatures:
        if not isinstance(signature, lattice_planes.PlaneSignature):
            raise V2AFailure("signature_replay_mismatch",
                             "signature collection contains a foreign object")
        signature_reading = signature.reading
        if not isinstance(signature_reading, lattice_planes.PlaneReading):
            raise V2AFailure("signature_replay_mismatch",
                             "a signature references a foreign reading")
        signature_key = (signature_reading.frame,
                         _reading_order_key_sha256(signature_reading))
        if signature_key != reading_key:
            continue
        alternatives = signature_reading.matching_alternatives(
            signature.blocks_per_period)
        if (len(alternatives) == 1
                and float(alternatives[0].blocks_per_period) == selected_b):
            matches.append(signature)
    if len(matches) != 1:
        raise V2AFailure("signature_replay_mismatch",
                         "the selected endpoint signature is missing or ambiguous")
    footprint = matches[0].region_footprint
    if footprint is None:
        return FOOTPRINT_ABSENT
    if not isinstance(footprint, lattice_planes.PlaneRegionFootprint):
        raise V2AFailure("signature_replay_mismatch",
                         "endpoint footprint has a foreign type")
    return _sha256_json(footprint.to_record())


def _gate_reason(c2rb_result: Any, vector_result: Any) -> str | None:
    if (not isinstance(c2rb_result, plane_relative_translation.C2RBResult)
            or not isinstance(vector_result,
                             plane_vector_domain.C2RVectorDomainResult)):
        return "c2rb_result_invalid"
    try:
        if c2rb_result.status != "complete":
            return "gate_not_ready"
        if c2rb_result.placement_readiness != "ready":
            return "gate_not_ready"
        if c2rb_result.connected is not True:
            return "gate_not_ready"
        if len(c2rb_result.components) != 1:
            return "gate_not_ready"
        component = c2rb_result.components[0]
        if component.outcome != "unique_validated":
            return "gate_not_ready"
        if component.cycle_rank < 1:
            return "gate_not_ready"
        if component.exhaustive is not True:
            return "gate_not_ready"
        if component.failing_chords:
            return "gate_not_ready"
        if any(cycle.matching_provenance_count <= 0
               for cycle in component.cycle_residuals):
            return "gate_not_ready"
        if len(component.translation_classes) != 1:
            return "gate_not_ready"
        if (vector_result.status != "complete" or vector_result.refusals
                or not vector_result.domains):
            return "gate_not_ready"
        if vector_result.represented_work != vector_result.planned_work:
            return "gate_not_ready"
    except (AttributeError, TypeError, ValueError, OverflowError):
        return "c2rb_result_invalid"
    return None


def _resolve_endpoint(
    ordered: Mapping[str, tuple[lattice_planes.PlaneReading, ...]],
    frame: str, plane_index: int, axis: int,
) -> lattice_planes.PlaneReading:
    values = ordered.get(frame)
    if not values or not 0 <= plane_index < len(values):
        raise V2AFailure("identity_replay_mismatch",
                         "canonical endpoint reading index cannot be resolved")
    reading = values[plane_index]
    if reading.frame != frame or reading.axis != axis:
        raise V2AFailure("identity_replay_mismatch",
                         "canonical endpoint object or axis differs")
    return reading


def _reading_identity(
    reading: lattice_planes.PlaneReading,
    lookup: Mapping[tuple[str, tuple[Any, ...]], int],
    universe_sha256: str,
) -> ReadingIdentity:
    index = _canonical_position(reading, lookup)
    key_sha256 = _reading_order_key_sha256(reading)
    source_region_id = (reading.region_evidence.source_region_id
                        if reading.region_evidence is not None else None)
    return ReadingIdentity.build(
        frame=reading.frame, canonical_index=index,
        reading_order_key_sha256=key_sha256, reading_axis=reading.axis,
        source_region_id=source_region_id,
        canonical_universe_sha256=universe_sha256)


def _delta_axis_map(
    domain: plane_vector_domain.MatchLateralVectorDomain, global_index: int,
) -> dict[int, float]:
    return {axis: float(value) for axis, value in domain.vector_at(global_index)}


def _candidate_feasible(
    domain: plane_vector_domain.MatchLateralVectorDomain,
    accepted_dyadic: Sequence[tuple[int, int]], edge_kind: str,
    closure: Fraction, global_index: int,
) -> bool:
    observed = _delta_axis_map(domain, global_index)
    normal_axis = domain.normal_constraint.world_axis
    observed[normal_axis] = float(domain.normal_constraint.delta_translation)
    for axis in (0, 1, 2):
        if axis not in observed:
            raise V2AFailure("accepted_delta_missing",
                             "A4 candidate lacks one world axis")
        difference = (_fraction(_dyadic_from_float(observed[axis]))
                      - _fraction(accepted_dyadic[axis]))
        if edge_kind == "tree":
            if difference != 0:
                return False
        elif abs(difference) > closure:
            return False
    return True


class _BoundPlan:
    """Validated, identity-bound V2-A inputs before candidate enumeration."""

    __slots__ = ("domains", "accepted", "references", "edge_endpoints",
                 "domain_by_reading_sha256", "reading_domains",
                 "total_candidates")

    def __init__(self, domains, accepted, references, edge_endpoints,
                 domain_by_reading_sha256, reading_domains, total_candidates):
        self.domains = domains
        self.accepted = accepted
        self.references = references
        self.edge_endpoints = edge_endpoints
        self.domain_by_reading_sha256 = domain_by_reading_sha256
        self.reading_domains = reading_domains
        self.total_candidates = total_candidates


def _bind_pipeline(
    c2rb_result: plane_relative_translation.C2RBResult,
    vector_result: plane_vector_domain.C2RVectorDomainResult,
    readings: Sequence[lattice_planes.PlaneReading],
    signatures: Sequence[lattice_planes.PlaneSignature],
) -> _BoundPlan:
    universe_sha256 = _canonical_universe_sha256(readings)
    if universe_sha256 != vector_result.canonical_reading_universe_sha256:
        raise V2AFailure("canonical_universe_mismatch",
                         "A4 canonical-universe digest does not replay")
    ordered = _ordered_readings(readings)
    lookup = _canonical_index_lookup(ordered)
    component = c2rb_result.components[0]
    certificate = component.translation_classes[0]
    translations = {frame: xyz for frame, xyz in
                    certificate.translations_dyadic_xyz}
    reference_by_identity = {reference.match_identity: reference
                             for reference in certificate.edge_references}
    chord_by_identity = {cycle.chord_identity: cycle
                         for cycle in component.cycle_residuals}
    tree_identities = set(component.tree_edges)

    domains = tuple(sorted(vector_result.domains,
                           key=lambda domain: domain.match_identity.sort_key()))
    accepted: dict[Any, tuple[str, tuple[tuple[int, int], ...]]] = {}
    for domain in domains:
        identity = domain.match_identity
        if identity in tree_identities:
            if (identity.left_frame not in translations
                    or identity.right_frame not in translations):
                raise V2AFailure("accepted_delta_missing",
                                 "tree endpoint lacks a certificate translation")
            delta = tuple(
                _sub_dyadic(translations[identity.right_frame][axis],
                            translations[identity.left_frame][axis])
                for axis in range(3))
            accepted[identity] = ("tree", delta)
        elif identity in chord_by_identity:
            cycle = chord_by_identity[identity]
            accepted[identity] = (
                "chord", tuple(cycle.required_chord_delta_dyadic_xyz))
        else:
            raise V2AFailure("graph_binding_mismatch",
                             "an A4 edge is neither a tree edge nor a chord")

    endpoint_identities: dict[ReadingIdentity, lattice_planes.PlaneReading] = {}
    atoms: dict[tuple[ReadingIdentity, str], dict[str, Any]] = {}
    edge_endpoints: dict[Any, tuple[ReadingIdentity, ReadingIdentity]] = {}
    footprint_cache: dict[tuple[tuple[str, str], float], str] = {}

    def _footprint(reading, selected_b):
        key = ((reading.frame, _reading_order_key_sha256(reading)),
               float(selected_b))
        value = footprint_cache.get(key)
        if value is None:
            value = _footprint_identity(reading, float(selected_b), signatures)
            footprint_cache[key] = value
        return value

    for domain in domains:
        identity = domain.match_identity
        left_reading = _resolve_endpoint(
            ordered, identity.left_frame, identity.left_plane, identity.left_axis)
        right_reading = _resolve_endpoint(
            ordered, identity.right_frame, identity.right_plane,
            identity.right_axis)
        left_identity = _reading_identity(left_reading, lookup, universe_sha256)
        right_identity = _reading_identity(right_reading, lookup, universe_sha256)
        edge_endpoints[identity] = (left_identity, right_identity)
        endpoint_identities[left_identity] = left_reading
        endpoint_identities[right_identity] = right_reading
        for factor in domain.axis_factors:
            if (plane_graph.axis_transform(factor.left_local_axis_label,
                                           factor.left_frame_quadrant)
                    != (factor.world_axis, factor.left_world_sign)
                    or plane_graph.axis_transform(factor.right_local_axis_label,
                                                  factor.right_frame_quadrant)
                    != (factor.world_axis, factor.right_world_sign)):
                raise V2AFailure("identity_replay_mismatch",
                                 "axis factor local labels do not replay")
            for reading, reading_identity, side, quadrant in (
                    (left_reading, left_identity, factor.left_side,
                     factor.left_frame_quadrant),
                    (right_reading, right_identity, factor.right_side,
                     factor.right_frame_quadrant)):
                atom = {
                    "record_sha256": side.record_sha256,
                    "selected_b": float(side.blocks_per_period),
                    "local_axis": side.local_axis,
                    "local_axis_label": side.local_axis_label,
                    "candidate_values": tuple(float(item)
                                              for item in side.candidate_values),
                    "absolute_quadrant": quadrant,
                    "footprint_identity": _footprint(
                        reading, float(side.blocks_per_period)),
                }
                key = (reading_identity, side.local_axis)
                existing = atoms.get(key)
                if existing is None:
                    atoms[key] = atom
                elif _sha256_json(existing) != _sha256_json(atom):
                    raise V2AFailure(
                        "endpoint_source_conflict",
                        "one reading and local axis carry two P1-B source atoms")

    reading_domains: list[ReadingLabelDomain] = []
    domain_by_reading_sha256: dict[str, ReadingLabelDomain] = {}
    for reading_identity in sorted(endpoint_identities,
                                   key=lambda item: item.sort_key()):
        atom_u = atoms.get((reading_identity, "u"))
        atom_v = atoms.get((reading_identity, "v"))
        if atom_u is None or atom_v is None:
            raise V2AFailure("reading_axis_missing",
                             "a reading endpoint has no local-axis source atom")
        if atom_u["selected_b"] != atom_v["selected_b"]:
            raise V2AFailure("endpoint_source_conflict",
                             "one reading pins two canonical b values")
        if atom_u["absolute_quadrant"] != atom_v["absolute_quadrant"]:
            raise V2AFailure("endpoint_source_conflict",
                             "one reading pins two absolute quadrants")
        if atom_u["footprint_identity"] != atom_v["footprint_identity"]:
            raise V2AFailure("endpoint_source_conflict",
                             "one reading pins two footprint identities")
        residue_u = ResidueAxisDomain.build(
            local_axis="u", local_axis_label=atom_u["local_axis_label"],
            selected_b=atom_u["selected_b"],
            source_record_sha256=atom_u["record_sha256"],
            candidate_values=atom_u["candidate_values"])
        residue_v = ResidueAxisDomain.build(
            local_axis="v", local_axis_label=atom_v["local_axis_label"],
            selected_b=atom_v["selected_b"],
            source_record_sha256=atom_v["record_sha256"],
            candidate_values=atom_v["candidate_values"])
        label_domain = ReadingLabelDomain.build(
            reading_identity=reading_identity,
            selected_b=atom_u["selected_b"],
            absolute_quadrant=atom_u["absolute_quadrant"],
            footprint_identity=atom_u["footprint_identity"],
            residue_u=residue_u, residue_v=residue_v)
        if label_domain.reading_sha256 in domain_by_reading_sha256:
            raise V2AFailure("identity_replay_mismatch",
                             "two endpoint readings share one typed identity")
        reading_domains.append(label_domain)
        domain_by_reading_sha256[label_domain.reading_sha256] = label_domain

    reading_domains.sort(key=lambda domain: domain.reading_identity.sort_key())
    return _BoundPlan(
        domains=domains, accepted=accepted, references=reference_by_identity,
        edge_endpoints=edge_endpoints,
        domain_by_reading_sha256=domain_by_reading_sha256,
        reading_domains=tuple(reading_domains),
        total_candidates=sum(domain.represented_candidate_count
                             for domain in domains))


def _resolve_index(
    domain: plane_vector_domain.MatchLateralVectorDomain,
    accepted_dyadic: Sequence[tuple[int, int]], edge_kind: str,
    closure: Fraction, left_domain: ReadingLabelDomain,
    right_domain: ReadingLabelDomain, global_index: int,
) -> tuple[int, int] | None:
    """Replay one A4 index and return its induced endpoint label pair.

    Returns ``None`` when the index is numerically infeasible.  Only the
    index-only A4 accessors are read; no candidate record is materialized.
    """
    observed = _delta_axis_map(domain, global_index)
    normal_axis = domain.normal_constraint.world_axis
    observed[normal_axis] = float(domain.normal_constraint.delta_translation)
    for axis in (0, 1, 2):
        if axis not in observed:
            raise V2AFailure("accepted_delta_missing",
                             "A4 candidate lacks one world axis")
        difference = (_fraction(_dyadic_from_float(observed[axis]))
                      - _fraction(accepted_dyadic[axis]))
        if edge_kind == "tree":
            if difference != 0:
                return None
        elif abs(difference) > closure:
            return None
    provenance = domain.provenance_key_at(global_index)
    left_indices: dict[str, int | None] = {"u": None, "v": None}
    right_indices: dict[str, int | None] = {"u": None, "v": None}
    for factor, item in zip(domain.axis_factors, provenance[1]):
        left_indices[factor.left_local_axis] = int(item[2])
        right_indices[factor.right_local_axis] = int(item[4])
    if (left_indices["u"] is None or left_indices["v"] is None
            or right_indices["u"] is None or right_indices["v"] is None):
        raise V2AFailure(
            "endpoint_source_conflict",
            "one endpoint lacks a complete local-axis pair")
    try:
        left_code = left_domain.code_for(left_indices["u"], left_indices["v"])
        right_code = right_domain.code_for(right_indices["u"], right_indices["v"])
    except IndexError as error:
        raise V2AFailure(
            "endpoint_source_conflict",
            "a candidate residue index is outside its reading label domain") \
            from error
    return left_code, right_code


def _bucket_pairs(
    domain: plane_vector_domain.MatchLateralVectorDomain,
    accepted_dyadic: Sequence[tuple[int, int]], edge_kind: str,
    closure: Fraction, left_domain: ReadingLabelDomain,
    right_domain: ReadingLabelDomain, counter: _RuntimeCounter,
) -> tuple[list[KernelBucket], int, str]:
    """Stream ``E_m`` in canonical order, retaining only per-bucket aggregates.

    No tuple/list/set of every feasible global index is ever retained: each
    bucket keeps exactly its first (canonical minimum) index and its distinct
    witness count, so memory is ``O(buckets)`` regardless of ``|E_m|``.  The
    complete ordered feasible transcript is folded into one rolling SHA-256.
    """
    factors = domain.axis_factors
    axis_span = (factors[1].left_side.candidate_count
                 * factors[1].right_side.candidate_count)
    grouped: dict[tuple[int, int], tuple[int, int]] = {}
    transcript = hashlib.sha256()
    feasible = 0
    for _entry, start, _stop in domain.iter_entry_groups():
        for left0 in range(factors[0].left_side.candidate_count):
            for right0 in range(factors[0].right_side.candidate_count):
                for left1 in range(factors[1].left_side.candidate_count):
                    for right1 in range(factors[1].right_side.candidate_count):
                        counter.edge_candidates_inspected += 1
                        counter.residues_inspected += 4
                        counter.check()
                        global_index = (
                            start
                            + (left0 * factors[0].right_side.candidate_count
                               + right0) * axis_span
                            + left1 * factors[1].right_side.candidate_count
                            + right1)
                        resolved = _resolve_index(
                            domain, accepted_dyadic, edge_kind, closure,
                            left_domain, right_domain, global_index)
                        if resolved is None:
                            continue
                        left_code, right_code = resolved
                        feasible += 1
                        prior = grouped.get((left_code, right_code))
                        if prior is None:
                            grouped[(left_code, right_code)] = (global_index, 1)
                        else:
                            if edge_kind == "tree":
                                raise V2AFailure(
                                    "tree_injectivity_violation",
                                    "an exact tree edge carries two enumerated "
                                    "indices for one endpoint label pair")
                            grouped[(left_code, right_code)] = (
                                prior[0], prior[1] + 1)
                        transcript.update(_canonical_json(
                            [global_index, left_code, right_code]).encode("utf-8"))
                        transcript.update(b"\n")
    buckets = _bucket_records(edge_kind, grouped)
    return buckets, feasible, transcript.hexdigest()


def _bucket_records(
    edge_kind: str, grouped: Mapping[tuple[int, int], tuple[int, int]],
) -> list[KernelBucket]:
    """Build canonical buckets from compact per-pair ``(index, count)`` aggregates.

    Enumeration-based tree injectivity: every enumerated feasible global index
    is grouped by the full endpoint label pair it induces, and an exact tree
    group must contain exactly one distinct index.  Two distinct tree indices
    under one label pair are the frozen ``tree_injectivity_violation``; the
    rounded A4 delta is never algebraically inverted.
    """
    buckets: list[KernelBucket] = []
    for (left_code, right_code), (first_index, distinct_count) in grouped.items():
        if edge_kind == "tree" and distinct_count != 1:
            raise V2AFailure(
                "tree_injectivity_violation",
                "an exact tree edge carries two enumerated indices for one "
                "endpoint label pair")
        buckets.append(KernelBucket.build(
            global_index=first_index, left_code=left_code,
            right_code=right_code, multiplicity=distinct_count))
    buckets.sort(key=lambda bucket: (bucket.left_code, bucket.right_code,
                                     bucket.global_index))
    return buckets


def _enumerate_pipeline(
    bound: _BoundPlan, closure_fraction: Fraction, counter: _RuntimeCounter,
) -> tuple[list[KernelEdge], list[EdgeFeasibleRecord],
           list["EnumerationCommitment"]]:
    kernel_edges: list[KernelEdge] = []
    edge_records: list[EdgeFeasibleRecord] = []
    commitments: list[EnumerationCommitment] = []
    for domain in bound.domains:
        identity = domain.match_identity
        kind, accepted_delta = bound.accepted[identity]
        left_identity, right_identity = bound.edge_endpoints[identity]
        left_domain = bound.domain_by_reading_sha256[left_identity.identity_sha256]
        right_domain = bound.domain_by_reading_sha256[
            right_identity.identity_sha256]
        buckets, feasible, transcript = _bucket_pairs(
            domain, accepted_delta, kind, closure_fraction,
            left_domain, right_domain, counter)
        reference = bound.references.get(identity)
        if reference is None:
            raise V2AFailure(
                "graph_binding_mismatch",
                "an accepted edge has no stored certificate reference")
        if reference.domain_sha256 != domain.domain_sha256:
            raise V2AFailure("c2rb_replay_mismatch",
                             "the stored reference binds a different A4 domain")
        if kind == "tree" and tuple(reference.delta_dyadic_xyz) != tuple(
                accepted_delta):
            raise V2AFailure(
                "stored_delta_echo_mismatch",
                "the stored edge delta does not echo T[right]-T[left]")
        if not _strict_int(reference.global_index) or \
                reference.global_index >= domain.represented_candidate_count:
            raise V2AFailure("stored_reference_not_feasible",
                             "the stored reference index is outside its domain")
        resolved_reference = _resolve_index(
            domain, accepted_delta, kind, closure_fraction, left_domain,
            right_domain, reference.global_index)
        if resolved_reference is None:
            raise V2AFailure(
                "stored_reference_not_feasible",
                "the stored reference is not an enumerated E_m witness")
        reference_left_code, reference_right_code = resolved_reference
        if not any(bucket.left_code == reference_left_code
                   and bucket.right_code == reference_right_code
                   for bucket in buckets):
            raise V2AFailure(
                "stored_reference_not_feasible",
                "the stored reference induces no retained E_m bucket")
        if kind == "tree" and any(
                bucket.multiplicity != 1 for bucket in buckets):
            raise V2AFailure(
                "tree_injectivity_violation",
                "an exact tree edge carries a multiplicity above one")
        aggregates_sha256 = _sha256_json([
            [bucket.left_code, bucket.right_code, bucket.multiplicity]
            for bucket in buckets])
        commitment = EnumerationCommitment.build(
            match_identity=identity, edge_kind=kind,
            domain_sha256=domain.domain_sha256,
            accepted_delta_dyadic_xyz=accepted_delta,
            left_reading_sha256=left_domain.reading_sha256,
            right_reading_sha256=right_domain.reading_sha256,
            inspected_candidates=domain.represented_candidate_count,
            residues_inspected=4 * domain.represented_candidate_count,
            feasible_count=feasible, bucket_count=len(buckets),
            bucket_aggregates_sha256=aggregates_sha256,
            stream_transcript_sha256=transcript,
            reference_global_index=reference.global_index,
            reference_left_code=reference_left_code,
            reference_right_code=reference_right_code,
            reference_seen=True)
        sentinel = KernelRepresentativeSentinel(
            global_index=reference.global_index,
            residue_indices=tuple(reference.residue_indices),
            provenance_sha256=_sha256_json(_json_value(reference.provenance_key)))
        kernel_edge = KernelEdge.build(
            match_identity=identity, edge_kind=kind,
            accepted_delta_dyadic_xyz=accepted_delta,
            left_reading_sha256=left_domain.reading_sha256,
            right_reading_sha256=right_domain.reading_sha256,
            buckets=buckets, ignored_representative=sentinel)
        kernel_edges.append(kernel_edge)
        edge_records.append(EdgeFeasibleRecord.build(
            match_identity=identity, edge_kind=kind,
            accepted_delta_dyadic_xyz=accepted_delta, feasible_count=feasible,
            label_pairs=tuple((bucket.left_code, bucket.right_code,
                               bucket.multiplicity) for bucket in buckets),
            stored_reference_global_index=reference.global_index,
            stored_reference_membership=True,
            stored_reference_left_code=reference_left_code,
            stored_reference_right_code=reference_right_code,
            domain_sha256=domain.domain_sha256,
            kernel_edge_sha256=kernel_edge.edge_sha256,
            enumeration_commitment_sha256=commitment.commitment_sha256,
            left_reading_sha256=left_domain.reading_sha256,
            right_reading_sha256=right_domain.reading_sha256,
            inspected_candidates=domain.represented_candidate_count))
        commitments.append(commitment)
    kernel_edges.sort(key=lambda edge: edge.match_identity.sort_key())
    edge_records.sort(key=lambda record: record.match_identity.sort_key())
    commitments.sort(key=lambda record: record.match_identity.sort_key())
    return kernel_edges, edge_records, commitments



def _enumeration_commitment_payload(
        value: "EnumerationCommitment") -> dict[str, Any]:
    return {
        "match_identity": value.match_identity.to_record(),
        "edge_kind": value.edge_kind,
        "domain_sha256": value.domain_sha256,
        "accepted_delta_dyadic_xyz": _dyadics_record(
            value.accepted_delta_dyadic_xyz),
        "left_reading_sha256": value.left_reading_sha256,
        "right_reading_sha256": value.right_reading_sha256,
        "inspected_candidates": value.inspected_candidates,
        "residues_inspected": value.residues_inspected,
        "feasible_count": value.feasible_count,
        "bucket_count": value.bucket_count,
        "bucket_aggregates_sha256": value.bucket_aggregates_sha256,
        "stream_transcript_sha256": value.stream_transcript_sha256,
        "reference_global_index": value.reference_global_index,
        "reference_left_code": value.reference_left_code,
        "reference_right_code": value.reference_right_code,
        "reference_seen": value.reference_seen,
    }


@dataclass(frozen=True, slots=True)
class EnumerationCommitment:
    """Independent per-edge streaming commitment over the complete ``E_m``.

    Computed during the canonical streaming enumeration from the accepted A4
    ``domain_sha256`` plus the rolling ordered feasible-tuple transcript and the
    canonical bucket aggregates and counts.  It is a sibling of (not a field of)
    ``EdgeFeasibleRecord``: resealing a mutated edge record, bucket aggregate, or
    counter never repairs a disagreement with this independently retained
    commitment.  It does not authenticate an adversary that consistently rebuilds
    every root.
    """

    match_identity: plane_axis_support.C1MatchIdentity
    edge_kind: str
    domain_sha256: str
    accepted_delta_dyadic_xyz: tuple[
        tuple[int, int], tuple[int, int], tuple[int, int]]
    left_reading_sha256: str
    right_reading_sha256: str
    inspected_candidates: int
    residues_inspected: int
    feasible_count: int
    bucket_count: int
    bucket_aggregates_sha256: str
    stream_transcript_sha256: str
    reference_global_index: int
    reference_left_code: int
    reference_right_code: int
    reference_seen: bool
    commitment_sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(self.match_identity,
                           plane_axis_support.C1MatchIdentity)
                or self.edge_kind not in EDGE_KINDS
                or not _valid_digest(self.domain_sha256)
                or type(self.accepted_delta_dyadic_xyz) is not tuple
                or len(self.accepted_delta_dyadic_xyz) != 3
                or any(not _validate_dyadic(item)
                       for item in self.accepted_delta_dyadic_xyz)
                or not _valid_digest(self.left_reading_sha256)
                or not _valid_digest(self.right_reading_sha256)
                or not _strict_int(self.inspected_candidates, minimum=1)
                or not _strict_int(self.residues_inspected)
                or self.residues_inspected != 4 * self.inspected_candidates
                or not _strict_int(self.feasible_count, minimum=1)
                or self.feasible_count > self.inspected_candidates
                or not _strict_int(self.bucket_count, minimum=1)
                or self.bucket_count > self.feasible_count
                or not _valid_digest(self.bucket_aggregates_sha256)
                or not _valid_digest(self.stream_transcript_sha256)
                or not _strict_int(self.reference_global_index)
                or self.reference_global_index >= self.inspected_candidates
                or not _strict_int(self.reference_left_code)
                or not _strict_int(self.reference_right_code)
                or type(self.reference_seen) is not bool
                or not _valid_digest(self.commitment_sha256)
                or _sha256_json(_enumeration_commitment_payload(self))
                != self.commitment_sha256):
            raise ValueError("V2-A enumeration commitment is malformed")

    @classmethod
    def build(cls, *, match_identity: plane_axis_support.C1MatchIdentity,
              edge_kind: str, domain_sha256: str,
              accepted_delta_dyadic_xyz: Sequence[tuple[int, int]],
              left_reading_sha256: str, right_reading_sha256: str,
              inspected_candidates: int, residues_inspected: int,
              feasible_count: int, bucket_count: int,
              bucket_aggregates_sha256: str, stream_transcript_sha256: str,
              reference_global_index: int, reference_left_code: int,
              reference_right_code: int,
              reference_seen: bool) -> "EnumerationCommitment":
        return _seal(cls, {
            "match_identity": match_identity, "edge_kind": edge_kind,
            "domain_sha256": domain_sha256,
            "accepted_delta_dyadic_xyz": tuple(
                tuple(item) for item in accepted_delta_dyadic_xyz),
            "left_reading_sha256": left_reading_sha256,
            "right_reading_sha256": right_reading_sha256,
            "inspected_candidates": inspected_candidates,
            "residues_inspected": residues_inspected,
            "feasible_count": feasible_count, "bucket_count": bucket_count,
            "bucket_aggregates_sha256": bucket_aggregates_sha256,
            "stream_transcript_sha256": stream_transcript_sha256,
            "reference_global_index": reference_global_index,
            "reference_left_code": reference_left_code,
            "reference_right_code": reference_right_code,
            "reference_seen": reference_seen,
        }, "commitment_sha256", _enumeration_commitment_payload)

    def to_record(self) -> dict[str, Any]:
        record = _enumeration_commitment_payload(self)
        record["commitment_sha256"] = self.commitment_sha256
        return record


def _edge_record_payload(value: "EdgeFeasibleRecord") -> dict[str, Any]:
    return {
        "match_identity": value.match_identity.to_record(),
        "edge_kind": value.edge_kind,
        "domain_sha256": value.domain_sha256,
        "accepted_delta_dyadic_xyz": _dyadics_record(
            value.accepted_delta_dyadic_xyz),
        "feasible_count": value.feasible_count,
        "label_pairs": [[left_code, right_code, multiplicity]
                        for left_code, right_code, multiplicity
                        in value.label_pairs],
        "stored_reference_global_index": value.stored_reference_global_index,
        "stored_reference_membership": value.stored_reference_membership,
        "stored_reference_left_code": value.stored_reference_left_code,
        "stored_reference_right_code": value.stored_reference_right_code,
        "kernel_edge_sha256": value.kernel_edge_sha256,
        "enumeration_commitment_sha256": value.enumeration_commitment_sha256,
        "left_reading_sha256": value.left_reading_sha256,
        "right_reading_sha256": value.right_reading_sha256,
        "inspected_candidates": value.inspected_candidates,
    }


@dataclass(frozen=True, slots=True)
class EdgeFeasibleRecord:
    """One edge's complete feasible-set summary and induced label-pair buckets."""

    match_identity: plane_axis_support.C1MatchIdentity
    edge_kind: str
    domain_sha256: str
    accepted_delta_dyadic_xyz: tuple[
        tuple[int, int], tuple[int, int], tuple[int, int]]
    feasible_count: int
    label_pairs: tuple[tuple[int, int, int], ...]
    stored_reference_global_index: int
    stored_reference_membership: bool
    stored_reference_left_code: int
    stored_reference_right_code: int
    kernel_edge_sha256: str
    enumeration_commitment_sha256: str
    left_reading_sha256: str
    right_reading_sha256: str
    inspected_candidates: int
    record_sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(self.match_identity,
                           plane_axis_support.C1MatchIdentity)
                or self.edge_kind not in EDGE_KINDS
                or not _valid_digest(self.domain_sha256)
                or type(self.accepted_delta_dyadic_xyz) is not tuple
                or len(self.accepted_delta_dyadic_xyz) != 3
                or any(not _validate_dyadic(value)
                       for value in self.accepted_delta_dyadic_xyz)
                or not _strict_int(self.feasible_count, minimum=1)
                or type(self.label_pairs) is not tuple or not self.label_pairs
                or any(type(item) is not tuple or len(item) != 3
                       or not _strict_int(item[0]) or not _strict_int(item[1])
                       or not _strict_int(item[2], minimum=1)
                       for item in self.label_pairs)
                or tuple(sorted(self.label_pairs)) != self.label_pairs
                or len(set((item[0], item[1])
                           for item in self.label_pairs)) != len(self.label_pairs)
                or sum(item[2] for item in self.label_pairs)
                != self.feasible_count
                or not _strict_int(self.stored_reference_global_index)
                or self.stored_reference_membership is not True
                or not _strict_int(self.stored_reference_left_code)
                or not _strict_int(self.stored_reference_right_code)
                or not _valid_digest(self.kernel_edge_sha256)
                or not _valid_digest(self.enumeration_commitment_sha256)
                or not _valid_digest(self.left_reading_sha256)
                or not _valid_digest(self.right_reading_sha256)
                or not _strict_int(self.inspected_candidates, minimum=1)
                or self.feasible_count > self.inspected_candidates
                or self.stored_reference_global_index
                >= self.inspected_candidates
                or self.edge_kind == "tree" and any(
                    item[2] != 1 for item in self.label_pairs)
                or (self.stored_reference_left_code,
                    self.stored_reference_right_code) not in {
                        (item[0], item[1]) for item in self.label_pairs}
                or not _valid_digest(self.record_sha256)
                or _sha256_json(_edge_record_payload(self)) != self.record_sha256):
            raise ValueError("V2-A edge feasible record is malformed")

    @classmethod
    def build(cls, *, match_identity: plane_axis_support.C1MatchIdentity,
              edge_kind: str, domain_sha256: str,
              accepted_delta_dyadic_xyz: Sequence[tuple[int, int]],
              feasible_count: int,
              label_pairs: Sequence[tuple[int, int, int]],
              stored_reference_global_index: int,
              stored_reference_membership: bool,
              stored_reference_left_code: int,
              stored_reference_right_code: int,
              kernel_edge_sha256: str, enumeration_commitment_sha256: str,
              left_reading_sha256: str,
              right_reading_sha256: str,
              inspected_candidates: int) -> "EdgeFeasibleRecord":
        return _seal(cls, {
            "match_identity": match_identity, "edge_kind": edge_kind,
            "domain_sha256": domain_sha256,
            "accepted_delta_dyadic_xyz": tuple(
                tuple(item) for item in accepted_delta_dyadic_xyz),
            "feasible_count": feasible_count,
            "label_pairs": tuple(tuple(item) for item in label_pairs),
            "stored_reference_global_index": stored_reference_global_index,
            "stored_reference_membership": stored_reference_membership,
            "stored_reference_left_code": stored_reference_left_code,
            "stored_reference_right_code": stored_reference_right_code,
            "kernel_edge_sha256": kernel_edge_sha256,
            "enumeration_commitment_sha256": enumeration_commitment_sha256,
            "left_reading_sha256": left_reading_sha256,
            "right_reading_sha256": right_reading_sha256,
            "inspected_candidates": inspected_candidates,
        }, "record_sha256", _edge_record_payload)

    @property
    def induced_label_pair_count(self) -> int:
        return len(self.label_pairs)

    def to_record(self) -> dict[str, Any]:
        record = _edge_record_payload(self)
        record["record_sha256"] = self.record_sha256
        return record


def _cross_link_mismatch(
    reading_domains: Sequence[ReadingLabelDomain],
    edge_records: Sequence[EdgeFeasibleRecord],
    enumeration_commitments: Sequence["EnumerationCommitment"],
    witnesses: Sequence[EndpointAssignmentWitness],
    *, endpoint_assignment_count: int, planned_work: V2AWork,
    runtime_work: V2AWork, budget: V2ABudget,
    counter: _RuntimeCounter | None = None,
) -> str | None:
    """Frozen 4.5 parent semantic cross-link validation.

    Returns ``None`` when every retained witness, reading label, edge record,
    enumeration-commitment sibling, reported count, and work counter is mutually
    consistent, or a typed detail string when it is not.  Resealing one mutated
    record's own digest never repairs a semantic disagreement with its
    cross-linked siblings, its reading domains, or its independently retained
    enumeration commitment.  This is a relation among cross-linked records, not
    an authentication of an adversary that consistently rebuilds every root.
    """
    def unit() -> None:
        if counter is not None:
            counter.result_validation_cross_links += 1
            counter.check()

    domains: dict[str, ReadingLabelDomain] = {}
    for domain in reading_domains:
        unit()
        if not isinstance(domain, ReadingLabelDomain):
            return "a parent reading domain has a foreign type"
        if _sha256_json(_reading_domain_payload(domain)) != domain.domain_sha256:
            return "a parent reading domain digest does not replay"
        if domain.reading_sha256 in domains:
            return "two parent reading domains share one typed identity"
        domains[domain.reading_sha256] = domain

    commitments: dict[str, EnumerationCommitment] = {}
    for commitment in enumeration_commitments:
        unit()
        if not isinstance(commitment, EnumerationCommitment):
            return "a parent enumeration commitment has a foreign type"
        if (_sha256_json(_enumeration_commitment_payload(commitment))
                != commitment.commitment_sha256):
            return "a parent enumeration commitment digest does not replay"
        if commitment.commitment_sha256 in commitments:
            return "two parent enumeration commitments share one digest"
        if (commitment.left_reading_sha256 not in domains
                or commitment.right_reading_sha256 not in domains):
            return "a parent enumeration commitment binds an absent reading domain"
        commitments[commitment.commitment_sha256] = commitment
    if len(commitments) != len(edge_records):
        return "the parent enumeration commitments do not cover every edge"

    records: dict[str, EdgeFeasibleRecord] = {}
    for record in edge_records:
        unit()
        if not isinstance(record, EdgeFeasibleRecord):
            return "a parent edge record has a foreign type"
        if _sha256_json(_edge_record_payload(record)) != record.record_sha256:
            return "a parent edge record digest does not replay"
        if record.kernel_edge_sha256 in records:
            return "two parent edge records share one kernel edge"
        commitment = commitments.get(record.enumeration_commitment_sha256)
        if commitment is None:
            return "a parent edge record references an absent commitment sibling"
        if (commitment.match_identity != record.match_identity
                or commitment.edge_kind != record.edge_kind
                or commitment.domain_sha256 != record.domain_sha256
                or tuple(commitment.accepted_delta_dyadic_xyz)
                != tuple(record.accepted_delta_dyadic_xyz)
                or commitment.left_reading_sha256 != record.left_reading_sha256
                or commitment.right_reading_sha256 != record.right_reading_sha256
                or commitment.inspected_candidates
                != record.inspected_candidates
                or commitment.residues_inspected
                != 4 * record.inspected_candidates
                or commitment.feasible_count != record.feasible_count
                or commitment.bucket_count != len(record.label_pairs)
                or commitment.reference_global_index
                != record.stored_reference_global_index
                or commitment.reference_left_code
                != record.stored_reference_left_code
                or commitment.reference_right_code
                != record.stored_reference_right_code
                or commitment.reference_seen
                != record.stored_reference_membership):
            return "a parent edge record disagrees with its commitment sibling"
        if commitment.bucket_aggregates_sha256 != _sha256_json(
                [[item[0], item[1], item[2]] for item in record.label_pairs]):
            return "a parent edge record bucket aggregate is not its commitment"
        if (record.left_reading_sha256 not in domains
                or record.right_reading_sha256 not in domains):
            return "a parent edge record binds an absent reading domain"
        left_domain = domains[record.left_reading_sha256]
        right_domain = domains[record.right_reading_sha256]
        if sum(item[2] for item in record.label_pairs) != record.feasible_count:
            return "a parent edge record multiplicity sum misses feasible_count"
        if record.feasible_count > record.inspected_candidates:
            return "a parent edge record feasible_count exceeds inspected work"
        if (record.stored_reference_left_code,
                record.stored_reference_right_code) not in {
                (item[0], item[1]) for item in record.label_pairs}:
            return "a parent edge record stored reference has no retained bucket"
        for left_code, right_code, multiplicity in record.label_pairs:
            unit()
            if not 0 <= left_code < left_domain.label_count:
                return "a parent edge record label pair is outside the left domain"
            if not 0 <= right_code < right_domain.label_count:
                return "a parent edge record label pair is outside the right domain"
            if multiplicity < 1:
                return "a parent edge record carries a non-positive multiplicity"
            if record.edge_kind == "tree" and multiplicity != 1:
                return "an exact tree edge record carries multiplicity above one"
        records[record.kernel_edge_sha256] = record

    if len({witness.witness_sha256 for witness in witnesses}) != len(witnesses):
        return "parent witnesses are not injective"
    for witness in witnesses:
        unit()
        if not isinstance(witness, EndpointAssignmentWitness):
            return "a parent witness has a foreign type"
        if _sha256_json(_witness_payload(witness)) != witness.witness_sha256:
            return "a parent witness digest does not replay"
        if len(witness.labels) != len(reading_domains):
            return "a parent witness does not label every reading"
        codes: dict[str, int] = {}
        label_order: list[tuple[Any, ...]] = []
        for reading_sha256, index_u, index_v, label_sha256 in witness.labels:
            unit()
            domain = domains.get(reading_sha256)
            if domain is None:
                return "a parent witness labels an absent reading identity"
            if reading_sha256 in codes:
                return "a parent witness labels one reading twice"
            try:
                expected = domain.label_for(index_u, index_v)
            except (IndexError, ValueError):
                return "a parent witness label is outside its reading domain"
            if expected.label_sha256 != label_sha256:
                return ("a parent witness label digest disagrees with its "
                        "reading domain")
            codes[reading_sha256] = domain.code_for(index_u, index_v)
            label_order.append(domain.reading_identity.sort_key())
        if label_order != sorted(label_order):
            return "a parent witness label order is not canonical"
        if len(witness.edge_multiplicities) != len(edge_records):
            return "a parent witness does not cover every edge"
        seen_edges: set[str] = set()
        for edge_sha256, multiplicity in witness.edge_multiplicities:
            unit()
            record = records.get(edge_sha256)
            if record is None:
                return "a parent witness references an absent edge record"
            if edge_sha256 in seen_edges:
                return "a parent witness references one edge twice"
            seen_edges.add(edge_sha256)
            left_code = codes.get(record.left_reading_sha256)
            right_code = codes.get(record.right_reading_sha256)
            if left_code is None or right_code is None:
                return "a parent witness cannot induce an edge label pair"
            pair = next((item for item in record.label_pairs
                         if item[0] == left_code and item[1] == right_code), None)
            if pair is None or pair[2] != multiplicity:
                return ("a parent witness multiplicity disagrees with its edge "
                        "record")
    unit()
    if (len(reading_domains) != runtime_work.readings
            or len(reading_domains) != planned_work.readings):
        return "parent reading counts disagree with planned/runtime work"
    if (len(edge_records) != runtime_work.edges
            or len(edge_records) != planned_work.edges):
        return "parent edge counts disagree with planned/runtime work"
    endpoint_labels = sum(domain.label_count for domain in reading_domains)
    if (endpoint_labels != runtime_work.endpoint_labels
            or endpoint_labels != planned_work.endpoint_labels):
        return "parent endpoint-label count disagrees with planned/runtime work"
    inspected = sum(record.inspected_candidates for record in edge_records)
    if (inspected != runtime_work.edge_candidates_inspected
            or inspected != planned_work.edge_candidates_inspected):
        return "parent inspected-candidate count disagrees with planned/runtime work"
    if runtime_work.edge_candidates_inspected * 4 != runtime_work.residues_inspected:
        return "parent residue count disagrees with inspected candidates"
    if planned_work.edge_candidates_inspected * 4 != planned_work.residues_inspected:
        return "planned residue count disagrees with inspected candidates"
    if len(witnesses) != endpoint_assignment_count:
        return "parent witness count disagrees with the reported assignment count"
    if runtime_work.distinct_assignments != endpoint_assignment_count:
        return "parent distinct-assignment counter disagrees with the count"
    if runtime_work.retained_witnesses != len(witnesses):
        return "parent retained-witness counter disagrees with the witnesses"
    if runtime_work.result_validation_cross_links \
            > planned_work.result_validation_cross_links:
        return "parent cross-link work exceeds its deterministic plan"
    for work, label in ((planned_work, "planned"), (runtime_work, "runtime")):
        if (work.readings > budget.max_readings
                or work.edges > budget.max_edges
                or work.endpoint_labels > budget.max_endpoint_labels
                or work.residues_inspected > budget.max_residues_inspected
                or work.edge_candidates_inspected
                > budget.max_edge_candidates_inspected
                or work.partial_label_assignments
                > budget.max_partial_label_assignments
                or work.consistency_checks > budget.max_consistency_checks
                or work.result_validation_cross_links
                > budget.max_result_validation_cross_links
                or work.distinct_assignments > budget.max_distinct_assignments
                or work.retained_witnesses > budget.max_retained_witnesses):
            return "parent %s work exceeds a request budget cap" % label
    return None


def _reconstructed_kernel_initial_work(
    reading_domains: Sequence[ReadingLabelDomain],
    enumeration_commitments: Sequence["EnumerationCommitment"],
) -> V2AWork:
    """Rebuild the kernel's pre-search work from retained evidence only.

    The candidate and residue totals come from the independently retained
    per-edge ``EnumerationCommitment`` siblings, never from the mutable kernel
    runtime record.  Every kernel-search, result and cross-link counter is reset
    to zero, exactly as the canonical kernel received it before search.
    """
    return V2AWork(
        readings=len(reading_domains),
        edges=len(enumeration_commitments),
        endpoint_labels=sum(domain.label_count for domain in reading_domains),
        residues_inspected=sum(commitment.residues_inspected
                               for commitment in enumeration_commitments),
        edge_candidates_inspected=sum(commitment.inspected_candidates
                                      for commitment in enumeration_commitments),
        partial_label_assignments=0, consistency_checks=0,
        result_validation_cross_links=0, distinct_assignments=0,
        retained_witnesses=0, candidate_materializations=0)


def _kernel_edge_cross_link_mismatch(
    reading_domains: Sequence[ReadingLabelDomain],
    edge_records: Sequence[EdgeFeasibleRecord],
    kernel_edges: Sequence[KernelEdge],
) -> str | None:
    """Cross-link every edge record to exactly one retained canonical kernel edge.

    The retained kernel edge is the byte-level CSP input; its digest already
    excludes the opaque representative sentinel.  Each edge record must reference
    exactly one retained kernel edge with identical identity, kind, accepted
    delta and endpoint readings, and its exact ``label_pairs`` projection must be
    that kernel edge's complete ordered bucket list.
    """
    if not isinstance(kernel_edges, Sequence):
        return "retained kernel edges are not a sequence"
    if any(not isinstance(edge, KernelEdge) for edge in kernel_edges):
        return "a retained kernel edge has a foreign type"
    keys = [edge.match_identity.sort_key() for edge in kernel_edges]
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        return "retained kernel edges are not canonically ordered"
    if len({edge.edge_sha256 for edge in kernel_edges}) != len(kernel_edges):
        return "retained kernel edges are not injective"
    domains: dict[str, ReadingLabelDomain] = {}
    for domain in reading_domains:
        domains[domain.reading_sha256] = domain
    by_sha: dict[str, KernelEdge] = {}
    for edge in kernel_edges:
        if (not isinstance(edge.match_identity,
                           plane_axis_support.C1MatchIdentity)
                or _sha256_json(_kernel_edge_payload(edge)) != edge.edge_sha256):
            return "a retained kernel edge digest does not replay"
        by_sha[edge.edge_sha256] = edge
    if len(kernel_edges) != len(edge_records):
        return "the retained kernel edges do not cover every edge record"
    for record in edge_records:
        edge = by_sha.get(record.kernel_edge_sha256)
        if edge is None:
            return "an edge record references an absent retained kernel edge"
        if (edge.match_identity != record.match_identity
                or edge.edge_kind != record.edge_kind
                or tuple(edge.accepted_delta_dyadic_xyz)
                != tuple(record.accepted_delta_dyadic_xyz)
                or edge.left_reading_sha256 != record.left_reading_sha256
                or edge.right_reading_sha256 != record.right_reading_sha256):
            return "an edge record disagrees with its retained kernel edge"
        if (edge.left_reading_sha256 not in domains
                or edge.right_reading_sha256 not in domains):
            return "a retained kernel edge binds an absent reading domain"
        projection = tuple((bucket.left_code, bucket.right_code,
                            bucket.multiplicity) for bucket in edge.buckets)
        if projection != record.label_pairs:
            return ("an edge record label-pair projection is not its retained "
                    "kernel edge")
    return None


def _kernel_replay_mismatch(result: "V2AEndpointResult") -> str | None:
    """Deterministically rerun the isolated kernel from retained evidence.

    The rerun consumes the retained ``ReadingLabelDomain`` records, the retained
    canonical ``KernelEdge`` records, the same request budget, and the
    enumeration initial work reconstructed from the independently retained
    ``EnumerationCommitment`` candidate/residue totals with every kernel-search,
    result and cross-link counter reset to zero.  The resulting ``KernelResult``
    must reproduce the parent kernel phase byte for byte: classification,
    exhaustiveness, lower-bound flag, distinct-assignment count, retained
    witnesses, runtime work and digest.  A caller that mutates any
    kernel-produced work field and reseals ``kernel_result_sha256`` to match the
    mutated projection still fails, because the digest is re-derived from a fresh
    execution over immutable retained inputs.  This is a kernel-level replay; it
    never recurses through ``V2AEndpointResult`` validation and never enumerates
    or materializes an A4 candidate.
    """
    initial_work = _reconstructed_kernel_initial_work(
        result.reading_domains, result.enumeration_commitments)
    try:
        replay = solve_endpoint_provenance_kernel(
            domains=result.reading_domains, edges=result.kernel_edges,
            budget=result.budget, initial_work=initial_work)
    except (ValueError, TypeError, AttributeError, IndexError,
            OverflowError, ZeroDivisionError, FloatingPointError):
        return "the retained kernel inputs cannot be replayed"
    parent_work = _work_with_cross_links(result.runtime_work, 0)
    if (replay.status != "complete"
            or replay.classification != result.classification
            or replay.exhaustive != result.exhaustive
            or replay.lower_bound_only != result.lower_bound_only
            or replay.distinct_assignment_count
            != result.endpoint_assignment_count
            or replay.witnesses != result.witnesses
            or replay.runtime_work != parent_work):
        return "the retained kernel result does not replay from retained evidence"
    if replay.result_sha256 != result.kernel_result_sha256:
        return "the retained kernel digest does not replay from retained evidence"
    return None


def _public_result_payload(value: "V2AEndpointResult") -> dict[str, Any]:
    return {
        "schema": V2A_SCHEMA,
        "status": value.status,
        "classification": value.classification,
        "exhaustive": value.exhaustive,
        "translation_class_count": value.translation_class_count,
        "endpoint_assignment_count": value.endpoint_assignment_count,
        "lower_bound_only": value.lower_bound_only,
        "reading_domains": [domain.to_record()
                            for domain in value.reading_domains],
        "edge_records": [record.to_record() for record in value.edge_records],
        "enumeration_commitments": [commitment.to_record()
                                    for commitment in
                                    value.enumeration_commitments],
        "kernel_edges": [edge.to_record() for edge in value.kernel_edges],
        "enumeration_root_sha256": value.enumeration_root_sha256,
        "witnesses": [witness.to_record() for witness in value.witnesses],
        "planned_work": value.planned_work.to_record(),
        "runtime_work": value.runtime_work.to_record(),
        "budget": value.budget.to_record(),
        "refusals": [refusal.to_record() for refusal in value.refusals],
        "budget_failure_reason": value.budget_failure_reason,
        "canonical_universe_sha256": value.canonical_universe_sha256,
        "c2rb_result_sha256": value.c2rb_result_sha256,
        "vector_result_sha256": value.vector_result_sha256,
        "kernel_result_sha256": value.kernel_result_sha256,
    }


@dataclass(frozen=True, slots=True)
class V2AEndpointResult:
    """The self-validating public V2-A outcome."""

    status: str
    classification: str | None
    exhaustive: bool
    translation_class_count: int
    endpoint_assignment_count: int
    lower_bound_only: bool
    reading_domains: tuple[ReadingLabelDomain, ...]
    edge_records: tuple[EdgeFeasibleRecord, ...]
    enumeration_commitments: tuple[EnumerationCommitment, ...]
    kernel_edges: tuple[KernelEdge, ...]
    enumeration_root_sha256: str
    witnesses: tuple[EndpointAssignmentWitness, ...]
    planned_work: V2AWork
    runtime_work: V2AWork
    budget: V2ABudget
    refusals: tuple[V2ARefusal, ...]
    budget_failure_reason: str | None
    canonical_universe_sha256: str
    c2rb_result_sha256: str
    vector_result_sha256: str
    kernel_result_sha256: str
    input_sha256: str
    result_sha256: str

    def __post_init__(self) -> None:
        if (self.status not in STATUSES or type(self.exhaustive) is not bool
                or not _strict_int(self.translation_class_count, minimum=0)
                or not _strict_int(self.endpoint_assignment_count, minimum=0)
                or type(self.lower_bound_only) is not bool
                or type(self.reading_domains) is not tuple
                or any(not isinstance(value, ReadingLabelDomain)
                       for value in self.reading_domains)
                or type(self.edge_records) is not tuple
                or any(not isinstance(value, EdgeFeasibleRecord)
                       for value in self.edge_records)
                or type(self.enumeration_commitments) is not tuple
                or any(not isinstance(value, EnumerationCommitment)
                       for value in self.enumeration_commitments)
                or type(self.kernel_edges) is not tuple
                or any(not isinstance(value, KernelEdge)
                       for value in self.kernel_edges)
                or not _valid_digest(self.enumeration_root_sha256)
                or type(self.witnesses) is not tuple
                or any(not isinstance(value, EndpointAssignmentWitness)
                       for value in self.witnesses)
                or not isinstance(self.planned_work, V2AWork)
                or not isinstance(self.runtime_work, V2AWork)
                or not isinstance(self.budget, V2ABudget)
                or type(self.refusals) is not tuple
                or any(not isinstance(value, V2ARefusal)
                       for value in self.refusals)
                or not _valid_digest(self.canonical_universe_sha256)
                or not _valid_digest(self.c2rb_result_sha256)
                or not _valid_digest(self.vector_result_sha256)
                or not _valid_digest(self.kernel_result_sha256)
                or not _valid_digest(self.input_sha256)
                or not _valid_digest(self.result_sha256)):
            raise ValueError("V2-A endpoint result is malformed")
        if self.status == "complete":
            if (self.classification not in OUTCOMES or self.refusals
                    or self.budget_failure_reason is not None):
                raise ValueError("complete V2-A result is not a decision")
            ordered = [domain.reading_identity.sort_key()
                       for domain in self.reading_domains]
            if ordered != sorted(ordered) or len(set(ordered)) != len(ordered):
                raise ValueError(
                    "V2-A reading domains are not canonically ordered")
            edge_keys = [record.match_identity.sort_key()
                         for record in self.edge_records]
            if (edge_keys != sorted(edge_keys)
                    or len(set(edge_keys)) != len(edge_keys)):
                raise ValueError("V2-A edge records are not canonically ordered")
            if self.translation_class_count != 1:
                raise ValueError(
                    "complete V2-A result requires one translation class")
            if self.classification == "endpoint_unique":
                valid = (self.exhaustive and not self.lower_bound_only
                         and self.endpoint_assignment_count == 1
                         and len(self.witnesses) == 1)
            elif self.classification == "endpoint_none":
                valid = (self.exhaustive and not self.lower_bound_only
                         and self.endpoint_assignment_count == 0
                         and not self.witnesses)
            else:
                valid = (not self.exhaustive and self.lower_bound_only
                         and self.endpoint_assignment_count >= 2
                         and len(self.witnesses)
                         == self.endpoint_assignment_count)
            if not valid:
                raise ValueError(
                    "V2-A public classification disagrees with its proof records")
        elif self.status == "refused":
            if (self.classification is not None or self.reading_domains
                    or self.edge_records or self.enumeration_commitments
                    or self.kernel_edges
                    or self.witnesses
                    or not self.refusals
                    or self.budget_failure_reason is not None
                    or self.enumeration_root_sha256 != "0" * DIGEST_LENGTH
                    or self.runtime_work != V2AWork.zero()
                    or self.planned_work != V2AWork.zero()
                    or self.translation_class_count != 0
                    or self.endpoint_assignment_count != 0
                    or self.lower_bound_only or self.exhaustive):
                raise ValueError("refused V2-A result must be atomic")
            if tuple(sorted(self.refusals, key=_refusal_key)) != self.refusals:
                raise ValueError("V2-A refusals are not deterministically ordered")
        else:
            if (self.classification is not None or self.reading_domains
                    or self.edge_records or self.enumeration_commitments
                    or self.kernel_edges
                    or self.witnesses
                    or self.refusals
                    or self.budget_failure_reason not in BUDGET_FAILURE_REASONS
                    or self.enumeration_root_sha256 != "0" * DIGEST_LENGTH
                    or self.translation_class_count != 0
                    or self.endpoint_assignment_count != 0
                    or self.lower_bound_only or self.exhaustive):
                raise ValueError("budget-exhausted V2-A result must be atomic")
        if _sha256_json(_public_result_payload(self)) != self.result_sha256:
            raise ValueError("V2-A result digest does not replay")
        if any(_sha256_json(_reading_domain_payload(domain)) != domain.domain_sha256
               for domain in self.reading_domains):
            raise ValueError("V2-A result carries a tampered reading domain")
        if any(_sha256_json(_edge_record_payload(record)) != record.record_sha256
               for record in self.edge_records):
            raise ValueError("V2-A result carries a tampered edge record")
        if any(_sha256_json(_witness_payload(witness)) != witness.witness_sha256
               for witness in self.witnesses):
            raise ValueError("V2-A result carries a tampered witness")
        if _sha256_json({
                "canonical_universe_sha256": self.canonical_universe_sha256,
                "c2rb_result_sha256": self.c2rb_result_sha256,
                "vector_result_sha256": self.vector_result_sha256,
                "budget": self.budget.to_record(),
        }) != self.input_sha256:
            raise ValueError("V2-A input digest does not replay its request")
        if self.status == "complete":
            commitment_keys = [commitment.match_identity.sort_key()
                               for commitment in self.enumeration_commitments]
            if (commitment_keys != sorted(commitment_keys)
                    or len(set(commitment_keys)) != len(commitment_keys)):
                raise ValueError(
                    "V2-A enumeration commitments are not canonically ordered")
            if _sha256_json([commitment.commitment_sha256
                             for commitment in self.enumeration_commitments]) \
                    != self.enumeration_root_sha256:
                raise ValueError("V2-A enumeration root does not replay")
            edge_link_mismatch = _kernel_edge_cross_link_mismatch(
                self.reading_domains, self.edge_records, self.kernel_edges)
            if edge_link_mismatch is not None:
                raise ValueError(
                    "V2-A result fails its retained-kernel-edge cross-link: "
                    + edge_link_mismatch)
            link_counter = _RuntimeCounter(self.budget)
            try:
                mismatch = _cross_link_mismatch(
                    self.reading_domains, self.edge_records,
                    self.enumeration_commitments, self.witnesses,
                    endpoint_assignment_count=self.endpoint_assignment_count,
                    planned_work=self.planned_work,
                    runtime_work=self.runtime_work, budget=self.budget,
                    counter=link_counter)
            except V2ABudgetStop:
                raise ValueError(
                    "V2-A result exceeds its cross-link budget") from None
            if mismatch is not None:
                raise ValueError(
                    "V2-A result fails 4.5 cross-link validation: " + mismatch)
            expected_cross_links = (
                link_counter.result_validation_cross_links
                + _kernel_replay_validation_units(len(self.kernel_edges)))
            if (expected_cross_links
                    != self.runtime_work.result_validation_cross_links):
                raise ValueError(
                    "V2-A result cross-link counter does not replay its work")
            kernel_mismatch = _kernel_replay_mismatch(self)
            if kernel_mismatch is not None:
                raise ValueError(
                    "V2-A result fails its kernel replay cross-link: "
                    + kernel_mismatch)

    @classmethod
    def build(cls, **fields: Any) -> "V2AEndpointResult":
        provisional = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        if provisional.status == "complete":
            object.__setattr__(
                provisional, "enumeration_root_sha256",
                _sha256_json([commitment.commitment_sha256
                              for commitment in
                              provisional.enumeration_commitments]))
        else:
            object.__setattr__(provisional, "enumeration_root_sha256",
                               "0" * DIGEST_LENGTH)
        object.__setattr__(provisional, "input_sha256", _sha256_json({
            "canonical_universe_sha256": provisional.canonical_universe_sha256,
            "c2rb_result_sha256": provisional.c2rb_result_sha256,
            "vector_result_sha256": provisional.vector_result_sha256,
            "budget": provisional.budget.to_record(),
        }))
        object.__setattr__(provisional, "result_sha256",
                           _sha256_json(_public_result_payload(provisional)))
        cls.__post_init__(provisional)
        return provisional

    def to_record(self) -> dict[str, Any]:
        record = _public_result_payload(self)
        record["input_sha256"] = self.input_sha256
        record["result_sha256"] = self.result_sha256
        return record

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.to_record(), sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")


def _placeholder(value: Any) -> str:
    if _valid_digest(value):
        return value
    return "0" * DIGEST_LENGTH


def _refused_public_result(
    reason: str, detail: str, *, budget: V2ABudget,
    c2rb_sha256: str, vector_sha256: str, universe_sha256: str,
) -> V2AEndpointResult:
    return V2AEndpointResult.build(
        status="refused", classification=None, exhaustive=False,
        translation_class_count=0, endpoint_assignment_count=0,
        lower_bound_only=False, reading_domains=(), edge_records=(),
        enumeration_commitments=(), kernel_edges=(), witnesses=(),
        planned_work=V2AWork.zero(), runtime_work=V2AWork.zero(),
        budget=budget, refusals=(V2ARefusal(reason, detail),),
        budget_failure_reason=None,
        canonical_universe_sha256=_placeholder(universe_sha256),
        c2rb_result_sha256=_placeholder(c2rb_sha256),
        vector_result_sha256=_placeholder(vector_sha256),
        kernel_result_sha256="0" * DIGEST_LENGTH)


def _budget_public_result(
    reason: str, *, budget: V2ABudget, planned: V2AWork,
    runtime: V2AWork, c2rb_sha256: str, vector_sha256: str,
    universe_sha256: str,
) -> V2AEndpointResult:
    return V2AEndpointResult.build(
        status="budget_exhausted", classification=None, exhaustive=False,
        translation_class_count=0, endpoint_assignment_count=0,
        lower_bound_only=False, reading_domains=(), edge_records=(),
        enumeration_commitments=(), kernel_edges=(), witnesses=(),
        planned_work=planned,
        runtime_work=runtime, budget=budget,
        refusals=(), budget_failure_reason=reason,
        canonical_universe_sha256=_placeholder(universe_sha256),
        c2rb_result_sha256=_placeholder(c2rb_sha256),
        vector_result_sha256=_placeholder(vector_sha256),
        kernel_result_sha256="0" * DIGEST_LENGTH)


def _preflight_exceeds(plan: V2AWork, budget: V2ABudget) -> bool:
    return (
        plan.readings > budget.max_readings
        or plan.edges > budget.max_edges
        or plan.endpoint_labels > budget.max_endpoint_labels
        or plan.residues_inspected > budget.max_residues_inspected
        or plan.edge_candidates_inspected
            > budget.max_edge_candidates_inspected
        or plan.result_validation_cross_links
            > budget.max_result_validation_cross_links
    )


def _kernel_replay_validation_units(edge_count: int) -> int:
    """Deterministic result-validation units charged for the kernel replay.

    Each retained kernel edge is cross-linked to its edge record (one unit) and
    replayed through the isolated kernel (one unit), plus a single unit for the
    replay transaction.  The replay's own search work is separately bounded by
    the same kernel budget, so the whole validation phase has a deterministic
    plan bound under ``max_result_validation_cross_links`` and never recurses
    through ``V2AEndpointResult`` validation.
    """
    return 2 * edge_count + 1


def _work_with_cross_links(work: V2AWork, units: int) -> V2AWork:
    return V2AWork(
        work.readings, work.edges, work.endpoint_labels,
        work.residues_inspected, work.edge_candidates_inspected,
        work.partial_label_assignments, work.consistency_checks, units,
        work.distinct_assignments, work.retained_witnesses,
        work.candidate_materializations)


def analyze_endpoint_provenance(
    c2rb_result: plane_relative_translation.C2RBResult,
    vector_result: plane_vector_domain.C2RVectorDomainResult,
    support: plane_axis_support.C1AxisSupport,
    p1d_result: plane_shift_domain.ShiftDomainResult,
    matches: Sequence[plane_graph.PlaneMatch],
    phase_evidence: Sequence[plane_residue.HypothesisPhaseShiftEvidence],
    *,
    readings: Sequence[lattice_planes.PlaneReading],
    signatures: Sequence[lattice_planes.PlaneSignature],
    comparison_tolerance: float,
    closure_tolerance: float,
    budget: V2ABudget,
) -> V2AEndpointResult:
    """Run the exact bounded V2-A endpoint-provenance CSP under the frozen gate."""
    if not isinstance(budget, V2ABudget):
        raise ValueError("an explicit valid V2ABudget is required")
    c2rb_sha256 = _placeholder(getattr(c2rb_result, "result_sha256", None))
    vector_sha256 = _placeholder(getattr(vector_result, "result_sha256", None))
    universe_sha256 = _placeholder(
        getattr(vector_result, "canonical_reading_universe_sha256", None))
    if (type(comparison_tolerance) not in (int, float)
            or not math.isfinite(float(comparison_tolerance))
            or float(comparison_tolerance) < 0.0
            or type(closure_tolerance) not in (int, float)
            or not math.isfinite(float(closure_tolerance))
            or float(closure_tolerance) < 0.0):
        return _refused_public_result(
            "input_record_invalid",
            "tolerances must be finite and non-negative",
            budget=budget, c2rb_sha256=c2rb_sha256,
            vector_sha256=vector_sha256, universe_sha256=universe_sha256)
    gate = _gate_reason(c2rb_result, vector_result)
    if gate is not None:
        return _refused_public_result(
            gate, "the frozen section 2 V2-A input gate is not satisfied",
            budget=budget, c2rb_sha256=c2rb_sha256,
            vector_sha256=vector_sha256, universe_sha256=universe_sha256)
    counter = _RuntimeCounter(budget)
    try:
        if not plane_relative_translation.validate_relative_translation_graph(
                c2rb_result, vector_result, support, p1d_result, matches,
                phase_evidence, readings=readings, signatures=signatures,
                comparison_tolerance=comparison_tolerance,
                closure_tolerance=closure_tolerance):
            return _refused_public_result(
                "c2rb_replay_mismatch",
                "C2-R-B does not replay against current accepted sources",
                budget=budget, c2rb_sha256=c2rb_sha256,
                vector_sha256=vector_sha256, universe_sha256=universe_sha256)
        if not plane_vector_domain.validate_lateral_vector_domain_inputs(
                vector_result, support, p1d_result, matches, phase_evidence,
                readings=readings, signatures=signatures,
                comparison_tolerance=comparison_tolerance):
            return _refused_public_result(
                "a4_replay_mismatch",
                "A4 does not replay against current accepted sources",
                budget=budget, c2rb_sha256=c2rb_sha256,
                vector_sha256=vector_sha256, universe_sha256=universe_sha256)
        reading_tuple = tuple(readings)
        signature_tuple = tuple(signatures)
        bound = _bind_pipeline(c2rb_result, vector_result, reading_tuple,
                               signature_tuple)
        universe_sha256 = _placeholder(
            vector_result.canonical_reading_universe_sha256)
        total_candidates = bound.total_candidates
        domain_count = len(bound.reading_domains)
        edge_count = len(bound.domains)
        upper_witnesses = min(budget.max_distinct_assignments,
                              budget.max_retained_witnesses)
        cross_link_plan = (
            domain_count + 2 * edge_count + total_candidates
            + upper_witnesses * (1 + domain_count + edge_count) + 1
            + _kernel_replay_validation_units(edge_count))
        plan = V2AWork(
            readings=domain_count, edges=edge_count,
            endpoint_labels=sum(domain.label_count
                                for domain in bound.reading_domains),
            residues_inspected=4 * total_candidates,
            edge_candidates_inspected=total_candidates,
            partial_label_assignments=0, consistency_checks=0,
            result_validation_cross_links=cross_link_plan,
            distinct_assignments=0, retained_witnesses=0,
            candidate_materializations=0)
        if _preflight_exceeds(plan, budget):
            return _budget_public_result(
                "planned_budget_exceeded", budget=budget, planned=plan,
                runtime=V2AWork.zero(), c2rb_sha256=c2rb_sha256,
                vector_sha256=vector_sha256, universe_sha256=universe_sha256)
        closure_fraction = Fraction.from_float(float(closure_tolerance))
        counter.readings = len(bound.reading_domains)
        counter.edges = len(bound.domains)
        counter.endpoint_labels = plan.endpoint_labels
        kernel_edges, edge_records, commitments = _enumerate_pipeline(
            bound, closure_fraction, counter)
        kernel = solve_endpoint_provenance_kernel(
            domains=bound.reading_domains, edges=kernel_edges, budget=budget,
            initial_work=counter.snapshot())
        if kernel.status == "refused":
            return _refused_public_result(
                "kernel_input_invalid",
                "the isolated CSP kernel rejected its typed inputs",
                budget=budget, c2rb_sha256=c2rb_sha256,
                vector_sha256=vector_sha256, universe_sha256=universe_sha256)
        if kernel.status == "budget_exhausted":
            return _budget_public_result(
                kernel.budget_failure_reason or "runtime_budget_exceeded",
                budget=budget, planned=plan, runtime=kernel.runtime_work,
                c2rb_sha256=c2rb_sha256, vector_sha256=vector_sha256,
                universe_sha256=universe_sha256)
        link_counter = _RuntimeCounter(budget)
        try:
            mismatch = _cross_link_mismatch(
                bound.reading_domains, edge_records, commitments,
                kernel.witnesses,
                endpoint_assignment_count=kernel.distinct_assignment_count,
                planned_work=plan, runtime_work=kernel.runtime_work,
                budget=budget, counter=link_counter)
        except V2ABudgetStop as stop:
            return _budget_public_result(
                stop.reason, budget=budget, planned=plan,
                runtime=_work_with_cross_links(
                    kernel.runtime_work,
                    link_counter.result_validation_cross_links),
                c2rb_sha256=c2rb_sha256, vector_sha256=vector_sha256,
                universe_sha256=universe_sha256)
        if mismatch is not None:
            return _refused_public_result(
                "result_validation_mismatch", mismatch,
                budget=budget, c2rb_sha256=c2rb_sha256,
                vector_sha256=vector_sha256, universe_sha256=universe_sha256)
        final_work = _work_with_cross_links(
            kernel.runtime_work,
            link_counter.result_validation_cross_links
            + _kernel_replay_validation_units(len(kernel_edges)))
        return V2AEndpointResult.build(
            status="complete", classification=kernel.classification,
            exhaustive=kernel.exhaustive,
            translation_class_count=len(
                c2rb_result.components[0].translation_classes),
            endpoint_assignment_count=kernel.distinct_assignment_count,
            lower_bound_only=kernel.lower_bound_only,
            reading_domains=tuple(bound.reading_domains),
            edge_records=tuple(edge_records),
            enumeration_commitments=tuple(commitments),
            kernel_edges=tuple(kernel_edges),
            witnesses=tuple(kernel.witnesses),
            planned_work=plan, runtime_work=final_work, budget=budget,
            refusals=(), budget_failure_reason=None,
            canonical_universe_sha256=universe_sha256,
            c2rb_result_sha256=c2rb_sha256,
            vector_result_sha256=vector_sha256,
            kernel_result_sha256=kernel.result_sha256)
    except V2AFailure as failure:
        return _refused_public_result(
            failure.reason, failure.detail, budget=budget,
            c2rb_sha256=c2rb_sha256, vector_sha256=vector_sha256,
            universe_sha256=universe_sha256)
    except V2ABudgetStop as stop:
        return _budget_public_result(
            stop.reason, budget=budget, planned=V2AWork.zero(),
            runtime=counter.snapshot(), c2rb_sha256=c2rb_sha256,
            vector_sha256=vector_sha256, universe_sha256=universe_sha256)
    except (AttributeError, TypeError, ValueError, KeyError, IndexError,
            OverflowError, ZeroDivisionError, FloatingPointError):
        return _refused_public_result(
            "input_record_invalid",
            "V2-A rejected a malformed accepted input record",
            budget=budget, c2rb_sha256=c2rb_sha256,
            vector_sha256=vector_sha256, universe_sha256=universe_sha256)


def validate_endpoint_provenance_result(
    result: V2AEndpointResult,
    c2rb_result: plane_relative_translation.C2RBResult,
    vector_result: plane_vector_domain.C2RVectorDomainResult,
    support: plane_axis_support.C1AxisSupport,
    p1d_result: plane_shift_domain.ShiftDomainResult,
    matches: Sequence[plane_graph.PlaneMatch],
    phase_evidence: Sequence[plane_residue.HypothesisPhaseShiftEvidence],
    *,
    readings: Sequence[lattice_planes.PlaneReading],
    signatures: Sequence[lattice_planes.PlaneSignature],
    comparison_tolerance: float,
    closure_tolerance: float,
) -> bool:
    """Rebuild from current accepted sources and compare canonical result bytes."""
    if (not isinstance(result, V2AEndpointResult)
            or result.status != "complete"):
        return False
    try:
        replay = analyze_endpoint_provenance(
            c2rb_result, vector_result, support, p1d_result, matches,
            phase_evidence, readings=readings, signatures=signatures,
            comparison_tolerance=comparison_tolerance,
            closure_tolerance=closure_tolerance, budget=result.budget)
        return (replay.status == "complete"
                and replay.to_json_bytes() == result.to_json_bytes())
    except (AttributeError, TypeError, ValueError, OverflowError,
            FloatingPointError, ZeroDivisionError):
        return False
