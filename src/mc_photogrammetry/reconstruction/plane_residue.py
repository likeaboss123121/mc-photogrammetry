"""Hypothesis-scoped periodic-energy residue candidates and relative shift.

This module preserves ambiguity from a reading's possible period index. It
does not fuse rectified windows across views or choose a block residue.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import math
from typing import Any, Mapping

from mc_photogrammetry.reconstruction import lattice_planes, plane_graph


RESIDUE_REFUSAL_REASONS: tuple[str, ...] = (
    "reading_missing",
    "reading_identity_mismatch",
    "alternative_not_retained",
    "phase_evidence_absent",
    "phase_axis_unavailable",
    "invalid_period_fraction",
    "nonfinite_derived_value",
    "candidate_set_invalid",
    "cell_shift_absent",
)

_B_RATIONALS = {
    float(value): Fraction(str(float(value)))
    for value in lattice_planes.DISTANCE_ALTERNATIVES
}
_PHASE_UNAVAILABLE_REASONS = frozenset({
    "insufficient_period", "empty_profile", "nonfinite_profile",
    "zero_resultant", "invalid_profile", "invalid_period",
})
_MATCH_SUPPORT_FIELDS = (
    "psr", "peak", "second_peak", "score", "left_structural_share",
    "right_structural_share", "overlap_cells",
)
RELATIVE_SHIFT_SUPPORT_SCHEMA = "mc-photogrammetry-relative-shift-support-v2"
RELATIVE_SHIFT_SUPPORT_VERSION = 2
_CANDIDATE_REFUSALS = frozenset({
    "phase_evidence_absent", "phase_axis_unavailable",
    "invalid_period_fraction", "nonfinite_derived_value",
    "candidate_set_invalid",
})


def _finite(value: Any) -> float | None:
    if type(value) not in (int, float):
        return None
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _psr_record(value: float | None) -> dict[str, Any]:
    return {"value": None if value == float("inf") else value,
            "positive_infinity": value == float("inf")}


def _psr_from_record(value: object) -> float | None:
    if (type(value) is not dict or set(value) != {"value", "positive_infinity"}
            or type(value["positive_infinity"]) is not bool):
        raise ValueError("relative shift psr tag is malformed")
    if value["positive_infinity"]:
        if value["value"] is not None:
            raise ValueError("positive-infinity psr must have a null value")
        return float("inf")
    if value["value"] is None:
        return None
    parsed = _finite(value["value"])
    if parsed is None:
        raise ValueError("finite psr tag contains a non-finite value")
    return parsed


def _source_id(value: Any) -> str | None:
    if value is None:
        return None
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        return None
    return value


def _candidate_values(leading: float, fraction: float, rational: Fraction) -> tuple[float, ...]:
    values = []
    for index in range(rational.denominator):
        value = (leading + fraction * float(rational)
                 + index / rational.denominator) % 1.0
        if value == 0.0 or value == 1.0:
            value = 0.0
        values.append(float(value))
    return tuple(sorted(set(values)))


def _canonical_supported_b(value: Any) -> float | None:
    parsed = _finite(value)
    if parsed is None or parsed <= 0.0:
        return None
    matches = []
    for supported in _B_RATIONALS:
        tolerance = (1e-9 + 4.0 * math.ulp(1.0)
                     * max(1.0, abs(parsed), abs(supported)))
        if abs(parsed - supported) <= tolerance:
            matches.append(supported)
    return matches[0] if len(matches) == 1 else None


@dataclass(frozen=True, slots=True)
class PhaseAxisQuality:
    """P1-A quality measurements for one local rectified direction."""

    response_magnitude: float | None
    profile_mass: float | None
    concentration: float | None
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        if (self.unavailable_reason is not None
                and (type(self.unavailable_reason) is not str
                     or self.unavailable_reason not in _PHASE_UNAVAILABLE_REASONS)):
            raise ValueError("phase-axis quality has an unknown unavailable reason")
        for name, value in (("response_magnitude", self.response_magnitude),
                            ("profile_mass", self.profile_mass),
                            ("concentration", self.concentration)):
            if (value is not None
                    and _finite(value) is None):
                raise ValueError(f"phase-axis {name} must be finite or absent")
        if self.response_magnitude is not None and self.response_magnitude < 0.0:
            raise ValueError("phase-axis response magnitude cannot be negative")
        if self.profile_mass is not None and self.profile_mass < 0.0:
            raise ValueError("phase-axis profile mass cannot be negative")
        if (self.concentration is not None
                and not 0.0 <= self.concentration <= 1.0):
            raise ValueError("phase-axis concentration must lie in [0, 1]")
        if self.unavailable_reason is None:
            if any(value is None for value in (
                    self.response_magnitude, self.profile_mass, self.concentration)):
                raise ValueError("available phase-axis quality requires all measurements")
            if (self.response_magnitude <= 1e-12 or self.profile_mass <= 1e-9
                    or self.response_magnitude > self.profile_mass
                    + 1e-9 * max(1.0, self.profile_mass)):
                raise ValueError("available phase-axis quality is inconsistent")
            if not math.isclose(
                    self.concentration,
                    self.response_magnitude / self.profile_mass,
                    rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("phase-axis concentration disagrees with response and mass")
        if (self.response_magnitude is not None and self.profile_mass is not None
                and self.response_magnitude > self.profile_mass
                + 1e-9 * max(1.0, self.profile_mass)):
            raise ValueError("phase-axis response cannot exceed profile mass")
        if (self.response_magnitude is not None and self.profile_mass is not None
                and self.profile_mass > 0.0 and self.concentration is not None
                and not math.isclose(
                    self.concentration, self.response_magnitude / self.profile_mass,
                    rel_tol=1e-9, abs_tol=1e-12)):
            raise ValueError("phase-axis concentration disagrees with response and mass")
        elif self.unavailable_reason in {"nonfinite_profile", "invalid_profile"}:
            if any(value is not None for value in (
                    self.response_magnitude, self.profile_mass, self.concentration)):
                raise ValueError("invalid phase-axis profile cannot carry measurements")
        elif self.unavailable_reason == "empty_profile":
            if self.profile_mass is None or self.profile_mass > 1e-9:
                raise ValueError("empty phase-axis profile requires negligible mass")
        elif self.unavailable_reason == "zero_resultant":
            if (self.response_magnitude is None or self.response_magnitude > 1e-12
                    or self.profile_mass is None or self.profile_mass <= 1e-9):
                raise ValueError("zero-resultant quality requires positive mass and zero response")

    def to_record(self) -> dict[str, Any]:
        return {
            "response_magnitude": self.response_magnitude,
            "profile_mass": self.profile_mass,
            "concentration": self.concentration,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True, slots=True)
class PeriodicResidueCandidates:
    """All block residues admitted by one reading, local axis and b label."""

    frame: str
    plane_index: int
    reading_axis: int
    blocks_per_period: float
    local_axis: str
    phase_fraction: float | None
    rational_numerator: int
    rational_denominator: int
    leading_cell_coordinate: float | None
    residue_candidates: tuple[float, ...]
    phase_error_amplification: float
    source_region_id: str | None
    quality: PhaseAxisQuality | None
    refusal: str | None = None
    refusal_detail: str | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.frame, str) or not self.frame
                or type(self.plane_index) is not int
                or self.plane_index < 0 or type(self.reading_axis) is not int
                or self.reading_axis not in (0, 1, 2)
                or type(self.blocks_per_period) not in (int, float)
                or _finite(self.blocks_per_period) is None
                or float(self.blocks_per_period) <= 0.0
                or self.local_axis not in ("u", "v")
                or type(self.rational_numerator) is not int
                or type(self.rational_denominator) is not int
                or self.rational_numerator <= 0 or self.rational_denominator <= 0
                or _finite(self.phase_error_amplification) is None
                or self.phase_error_amplification < 0.0):
            raise ValueError("periodic residue identity or rational is invalid")
        b_value = float(self.blocks_per_period)
        expected_rational = _B_RATIONALS.get(b_value)
        if (expected_rational is None
                or (self.rational_numerator, self.rational_denominator)
                != (expected_rational.numerator, expected_rational.denominator)):
            raise ValueError("periodic residue rational must match a supported canonical b")
        if self.phase_error_amplification != abs(b_value):
            raise ValueError("phase-error amplification must equal abs(b)")
        if (self.refusal is not None
                and (type(self.refusal) is not str or self.refusal not in _CANDIDATE_REFUSALS)):
            raise ValueError("candidate refusal is not valid at the per-axis level")
        if self.source_region_id is not None and _source_id(self.source_region_id) is None:
            raise ValueError("periodic residue source id is invalid")
        if self.quality is not None and not isinstance(self.quality, PhaseAxisQuality):
            raise ValueError("periodic residue quality has an invalid type")
        if self.refusal is not None:
            if (not isinstance(self.residue_candidates, tuple) or self.residue_candidates
                    or self.phase_fraction is not None
                    or self.leading_cell_coordinate is not None):
                raise ValueError("refused residue records cannot contain candidates or derived values")
            if self.refusal == "phase_evidence_absent":
                if self.quality is not None or self.refusal_detail is not None:
                    raise ValueError("absent phase evidence cannot carry quality or detail")
            elif self.refusal == "phase_axis_unavailable":
                if (self.quality is None
                        or self.quality.unavailable_reason not in _PHASE_UNAVAILABLE_REASONS
                        or self.refusal_detail != self.quality.unavailable_reason):
                    raise ValueError("phase-axis refusal must preserve its exact P1-A reason")
            else:
                if self.refusal_detail is not None:
                    raise ValueError("only phase-axis refusal carries a refusal detail")
                if (self.quality is not None
                        and self.quality.unavailable_reason is not None):
                    raise ValueError("candidate refusal disagrees with unavailable phase quality")
        else:
            if (self.quality is None or self.quality.unavailable_reason is not None
                    or self.refusal_detail is not None
                    or self.phase_fraction is None or self.leading_cell_coordinate is None
                    or type(self.phase_fraction) not in (int, float)
                    or type(self.leading_cell_coordinate) not in (int, float)
                    or _finite(self.phase_fraction) is None
                    or not (0.0 <= self.phase_fraction < 1.0)
                    or _finite(self.leading_cell_coordinate) is None
                    or not isinstance(self.residue_candidates, tuple)
                    or len(self.residue_candidates) != self.rational_denominator):
                raise ValueError("available residue candidate set is incomplete")
            if (not isinstance(self.residue_candidates, tuple)
                    or any(_finite(value) is None
                           or not (0.0 <= value < 1.0)
                           for value in self.residue_candidates)):
                raise ValueError("residue candidates must be finite in [0, 1)")
            if tuple(sorted(set(self.residue_candidates))) != self.residue_candidates:
                raise ValueError("residue candidates must be sorted and unique")
            rational = expected_rational
            expected = _candidate_values(
                float(self.leading_cell_coordinate), float(self.phase_fraction), rational)
            if (len(expected) != rational.denominator
                    or self.residue_candidates != expected):
                raise ValueError("residue candidates disagree with the canonical formula")

    def to_record(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "plane_index": self.plane_index,
            "reading_axis": self.reading_axis,
            "blocks_per_period": float(self.blocks_per_period),
            "local_axis": self.local_axis,
            "phase_fraction": self.phase_fraction,
            "rational": {
                "numerator": self.rational_numerator,
                "denominator": self.rational_denominator,
            },
            "leading_cell_coordinate": self.leading_cell_coordinate,
            "residue_candidates": list(self.residue_candidates),
            "phase_error_amplification": float(self.phase_error_amplification),
            "source_region_id": self.source_region_id,
            "quality": None if self.quality is None else self.quality.to_record(),
            "refusal": self.refusal,
            "refusal_detail": self.refusal_detail,
        }


@dataclass(frozen=True, slots=True)
class RelativeShiftSupport:
    """Unmodified admitted cell shift and existing match support diagnostics."""

    schema: str
    version: int
    correlation_algorithm_id: str
    left_frame: str
    right_frame: str
    left_plane: int
    right_plane: int
    left_axis: int
    right_axis: int
    left_b: float | None
    right_b: float | None
    quadrant_of_right: int
    cell_shift_rc: tuple[int, int] | None
    psr: float | None
    peak: float | None
    second_peak: float | None
    overlap_cells: int | None
    left_structural_share: float | None
    right_structural_share: float | None
    score: float | None
    nonfinite_fields: tuple[str, ...] = ()
    refusal: str | None = None

    def __post_init__(self) -> None:
        if (self.schema != RELATIVE_SHIFT_SUPPORT_SCHEMA
                or type(self.version) is not int
                or self.version != RELATIVE_SHIFT_SUPPORT_VERSION
                or self.correlation_algorithm_id
                != plane_graph.PLANE_CORRELATION_ALGORITHM):
            raise ValueError("relative shift support provenance is unsupported")
        if (not isinstance(self.left_frame, str) or not self.left_frame
                or not isinstance(self.right_frame, str) or not self.right_frame
                or type(self.left_plane) is not int or self.left_plane < 0
                or type(self.right_plane) is not int or self.right_plane < 0
                or type(self.left_axis) is not int or self.left_axis not in (0, 1, 2)
                or type(self.right_axis) is not int or self.right_axis not in (0, 1, 2)
                or type(self.quadrant_of_right) is not int
                or self.quadrant_of_right not in (0, 1, 2, 3)):
            raise ValueError("relative shift identity is invalid")
        if self.cell_shift_rc is not None and (
                not isinstance(self.cell_shift_rc, tuple)
                or len(self.cell_shift_rc) != 2
                or any(type(value) is not int for value in self.cell_shift_rc)):
            raise ValueError("relative shift must be an exact (row, column) integer pair")
        if (self.refusal is not None
                and (type(self.refusal) is not str
                     or self.refusal not in RESIDUE_REFUSAL_REASONS)):
            raise ValueError("relative shift refusal is unknown")
        for value in (self.left_b, self.right_b):
            if (value is not None
                    and (_finite(value) is None or value <= 0.0)):
                raise ValueError("relative shift b values must be finite and positive or absent")
        if self.psr is not None and self.psr == float("-inf"):
            raise ValueError("relative shift psr cannot be negative infinity")
        for value in (self.peak, self.second_peak, self.score):
            if value is not None and _finite(value) is None:
                raise ValueError("relative shift support fields must be finite or absent")
        if (self.psr is not None and self.psr != float("inf")
                and _finite(self.psr) is None):
            raise ValueError("relative shift psr must be finite, positive infinity or absent")
        for value in (self.left_structural_share, self.right_structural_share):
            if (value is not None
                    and (_finite(value) is None or not 0.0 <= value <= 1.0)):
                raise ValueError("relative shift structural shares must be finite in [0, 1]")
        if self.overlap_cells is not None and (
                type(self.overlap_cells) is not int or self.overlap_cells < 0):
            raise ValueError("relative shift overlap must be a nonnegative integer or absent")
        if (not isinstance(self.nonfinite_fields, tuple)
                or any(type(name) is not str or name not in _MATCH_SUPPORT_FIELDS
                       for name in self.nonfinite_fields)
                or len(set(self.nonfinite_fields)) != len(self.nonfinite_fields)
                or tuple(name for name in _MATCH_SUPPORT_FIELDS
                         if name in self.nonfinite_fields) != self.nonfinite_fields):
            raise ValueError("relative shift nonfinite field list is malformed")
        support_values = {
            "psr": self.psr, "peak": self.peak,
            "second_peak": self.second_peak, "score": self.score,
            "left_structural_share": self.left_structural_share,
            "right_structural_share": self.right_structural_share,
            "overlap_cells": self.overlap_cells,
        }
        if any(support_values[name] is not None for name in self.nonfinite_fields):
            raise ValueError("nonfinite support fields must be absent in the record")
        expected_refusal = (
            "nonfinite_derived_value" if self.nonfinite_fields else
            "cell_shift_absent" if self.cell_shift_rc is None else None
        )
        if self.refusal != expected_refusal:
            raise ValueError("relative shift refusal disagrees with its support and shift")

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "correlation_algorithm_id": self.correlation_algorithm_id,
            "identity": {
                "left_frame": self.left_frame,
                "right_frame": self.right_frame,
                "left_plane": self.left_plane,
                "right_plane": self.right_plane,
                "left_axis": self.left_axis,
                "right_axis": self.right_axis,
                "left_b": self.left_b,
                "right_b": self.right_b,
                "quadrant_of_right": self.quadrant_of_right,
            },
            "cell_shift_rc": (None if self.cell_shift_rc is None
                              else list(self.cell_shift_rc)),
            "psr": _psr_record(self.psr),
            "peak": self.peak,
            "second_peak": self.second_peak,
            "overlap_cells": self.overlap_cells,
            "left_structural_share": self.left_structural_share,
            "right_structural_share": self.right_structural_share,
            "score": self.score,
            "nonfinite_fields": list(self.nonfinite_fields),
            "refusal": self.refusal,
        }


@dataclass(frozen=True, slots=True)
class HypothesisPhaseShiftEvidence:
    """Four local phase candidate sets plus one independent relative shift."""

    left_frame: str
    right_frame: str
    left_plane: int
    right_plane: int
    left_axis: int
    right_axis: int
    left_b: float | None
    right_b: float | None
    quadrant_of_right: int
    phase_candidates: tuple[PeriodicResidueCandidates, ...]
    relative_shift: RelativeShiftSupport
    refusal: str | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.left_frame, str) or not self.left_frame
                or not isinstance(self.right_frame, str) or not self.right_frame
                or type(self.left_plane) is not int or self.left_plane < 0
                or type(self.right_plane) is not int or self.right_plane < 0
                or type(self.left_axis) is not int or self.left_axis not in (0, 1, 2)
                or type(self.right_axis) is not int or self.right_axis not in (0, 1, 2)
                or type(self.quadrant_of_right) is not int
                or self.quadrant_of_right not in (0, 1, 2, 3)):
            raise ValueError("hypothesis identity is invalid")
        if (self.refusal is not None
                and (type(self.refusal) is not str
                     or self.refusal not in RESIDUE_REFUSAL_REASONS)):
            raise ValueError("hypothesis refusal is unknown")
        if (not isinstance(self.phase_candidates, tuple)
                or not isinstance(self.relative_shift, RelativeShiftSupport)):
            raise ValueError("hypothesis requires typed phase and relative-shift records")
        if self.refusal is None:
            if len(self.phase_candidates) != 4:
                raise ValueError("valid hypothesis evidence requires four local phase records")
            if self.left_b is None or self.right_b is None:
                raise ValueError("valid hypothesis evidence requires a finite b pair")
        elif self.phase_candidates:
            raise ValueError("refused top-level evidence cannot carry phase candidates")
        expected = (
            (self.left_frame, self.left_plane, self.left_axis, "u",
             _canonical_supported_b(self.left_b)),
            (self.left_frame, self.left_plane, self.left_axis, "v",
             _canonical_supported_b(self.left_b)),
            (self.right_frame, self.right_plane, self.right_axis, "u",
             _canonical_supported_b(self.right_b)),
            (self.right_frame, self.right_plane, self.right_axis, "v",
             _canonical_supported_b(self.right_b)),
        )
        actual = tuple((record.frame, record.plane_index, record.reading_axis,
                        record.local_axis, record.blocks_per_period)
                       for record in self.phase_candidates)
        if self.phase_candidates and actual != expected:
            raise ValueError("phase records are not bound to the hypothesis identity and b pair")
        if (self.relative_shift.left_frame != self.left_frame
                or self.relative_shift.right_frame != self.right_frame
                or self.relative_shift.left_plane != self.left_plane
                or self.relative_shift.right_plane != self.right_plane
                or self.relative_shift.left_axis != self.left_axis
                or self.relative_shift.right_axis != self.right_axis
                or self.relative_shift.left_b != self.left_b
                or self.relative_shift.right_b != self.right_b
                or self.relative_shift.quadrant_of_right != self.quadrant_of_right):
            raise ValueError("relative shift identity is not bound to the hypothesis")
        if self.refusal is None and any(value is None for value in (
                _canonical_supported_b(self.left_b),
                _canonical_supported_b(self.right_b))):
            raise ValueError("valid hypothesis b values must be supported alternatives")
        for value in (self.left_b, self.right_b):
            if value is not None and _finite(value) is None:
                raise ValueError("hypothesis b values must be finite or absent")

    def to_record(self) -> dict[str, Any]:
        record = {
            "identity": {
                "left_frame": self.left_frame,
                "right_frame": self.right_frame,
                "left_plane": self.left_plane,
                "right_plane": self.right_plane,
                "left_axis": self.left_axis,
                "right_axis": self.right_axis,
                "left_b": self.left_b,
                "right_b": self.right_b,
                "quadrant_of_right": self.quadrant_of_right,
            },
            "phase_candidates": [value.to_record() for value in self.phase_candidates],
            "relative_shift": self.relative_shift.to_record(),
            "refusal": self.refusal,
        }
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return record

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.to_record(), sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")


def _rational_b(value: float) -> Fraction | None:
    canonical = _B_RATIONALS.get(float(value))
    return canonical if canonical is not None else None


def _candidate_record(
    reading: lattice_planes.PlaneReading,
    plane_index: int,
    b_value: float,
    alternative: lattice_planes.DistanceAlternative,
    local_axis_index: int,
) -> PeriodicResidueCandidates:
    axis_name = "u" if local_axis_index == 0 else "v"
    rational = _rational_b(b_value)
    quality: PhaseAxisQuality | None = None
    phase_evidence = reading.phase_evidence
    source = (phase_evidence.source_region_id
              if phase_evidence is not None else None)
    distance = _finite(alternative.distance_blocks)
    period = _finite(reading.period_pixels)
    try:
        centre = (_finite(reading.rectification.centre[local_axis_index])
                  if len(reading.rectification.centre) == 2 else None)
    except (AttributeError, IndexError, TypeError):
        centre = None
    extent = _finite(reading.extent)
    amplification = abs(float(b_value))
    if rational is None:
        raise ValueError("candidate factory requires a supported canonical b")
    if phase_evidence is None:
        return PeriodicResidueCandidates(
            reading.frame, plane_index, int(reading.axis), b_value, axis_name, None,
            rational.numerator, rational.denominator, None, (), amplification,
            source, None, refusal="phase_evidence_absent")

    quality = PhaseAxisQuality(
        phase_evidence.response_magnitude_uv[local_axis_index],
        phase_evidence.profile_mass_uv[local_axis_index],
        phase_evidence.concentration_uv[local_axis_index],
        phase_evidence.unavailable_reason_uv[local_axis_index],
    )
    phase = phase_evidence.phase_uv_pixels[local_axis_index]
    unavailable = phase_evidence.unavailable_reason_uv[local_axis_index]
    if unavailable is not None:
        return PeriodicResidueCandidates(
            reading.frame, plane_index, int(reading.axis), b_value, axis_name, None,
            rational.numerator, rational.denominator, None, (), amplification,
            source, quality, refusal="phase_axis_unavailable",
            refusal_detail=unavailable)
    if phase is None:
        # Upstream P1-A records cannot have this combination. If a tampered
        # record bypasses that constructor, report a typed malformed-evidence
        # refusal instead of inventing an unavailable reason.
        return PeriodicResidueCandidates(
            reading.frame, plane_index, int(reading.axis), b_value, axis_name, None,
            rational.numerator, rational.denominator, None, (), amplification,
            source, quality, refusal="candidate_set_invalid")
    if (period is None or period <= 0.0 or distance is None or distance <= 0.0
            or centre is None or extent is None or extent <= 0.0):
        return PeriodicResidueCandidates(
            reading.frame, plane_index, int(reading.axis), b_value, axis_name, None,
            rational.numerator, rational.denominator, None, (), amplification,
            source, quality, refusal="nonfinite_derived_value")

    phase_value = _finite(phase)
    if phase_value is None or not (0.0 <= phase_value < period):
        return PeriodicResidueCandidates(
            reading.frame, plane_index, int(reading.axis), b_value, axis_name, None,
            rational.numerator, rational.denominator, None, (), amplification,
            source, quality, refusal="invalid_period_fraction")
    fraction = phase_value / period
    leading = distance * (centre - extent)
    base = leading + fraction * float(rational)
    if not math.isfinite(fraction) or not math.isfinite(leading) or not math.isfinite(base):
        return PeriodicResidueCandidates(
            reading.frame, plane_index, int(reading.axis), b_value, axis_name, None,
            rational.numerator, rational.denominator, None, (), amplification,
            source, quality, refusal="nonfinite_derived_value")

    canonical = _candidate_values(float(leading), float(fraction), rational)
    if (len(canonical) != rational.denominator
            or any(not math.isfinite(value) or not 0.0 <= value < 1.0
                   for value in canonical)):
        return PeriodicResidueCandidates(
            reading.frame, plane_index, int(reading.axis), b_value, axis_name, None,
            rational.numerator, rational.denominator, None, (), amplification,
            source, quality, refusal="candidate_set_invalid")
    return PeriodicResidueCandidates(
        reading.frame, plane_index, int(reading.axis), float(b_value), axis_name, float(fraction),
        rational.numerator, rational.denominator, float(leading), canonical,
        amplification, source, quality)


def _shift_support(
    match: plane_graph.PlaneMatch,
    left: lattice_planes.PlaneReading | None,
    right: lattice_planes.PlaneReading | None,
) -> RelativeShiftSupport:
    values: dict[str, float | None] = {}
    nonfinite: list[str] = []
    for name in ("psr", "peak", "second_peak", "score"):
        raw = getattr(match, name)
        parsed = (float("inf") if name == "psr" and raw == float("inf")
                  else _finite(raw))
        values[name] = parsed
        if parsed is None:
            nonfinite.append(name)
    left_share = _finite(left.structural_share) if left is not None else None
    right_share = _finite(right.structural_share) if right is not None else None
    if left is not None and left_share is None:
        nonfinite.append("left_structural_share")
    if right is not None and right_share is None:
        nonfinite.append("right_structural_share")
    overlap = (match.overlap_cells
               if type(match.overlap_cells) is int and match.overlap_cells >= 0
               else None)
    if overlap is None:
        nonfinite.append("overlap_cells")
    refusal = ("nonfinite_derived_value" if nonfinite else
               "cell_shift_absent" if match.cell_shift_rc is None else None)
    return RelativeShiftSupport(
        RELATIVE_SHIFT_SUPPORT_SCHEMA, RELATIVE_SHIFT_SUPPORT_VERSION,
        plane_graph.PLANE_CORRELATION_ALGORITHM,
        str(match.left), str(match.right), int(match.left_plane), int(match.right_plane),
        int(match.left_axis), int(match.right_axis), _finite(match.left_b),
        _finite(match.right_b), int(match.quadrant_of_right), match.cell_shift_rc,
        values["psr"], values["peak"], values["second_peak"], overlap,
        left_share, right_share, values["score"], tuple(nonfinite), refusal)


def hypothesis_phase_shift_evidence(
    match: plane_graph.PlaneMatch,
    readings: Mapping[tuple[str, int], lattice_planes.PlaneReading],
) -> HypothesisPhaseShiftEvidence:
    """Return four local residue candidate sets and copied relative match support."""
    if not isinstance(match, plane_graph.PlaneMatch):
        raise TypeError("match must be a PlaneMatch")
    left_b = _finite(match.left_b)
    right_b = _finite(match.right_b)
    left_key = (match.left, match.left_plane)
    right_key = (match.right, match.right_plane)
    left = readings.get(left_key)
    right = readings.get(right_key)
    left_valid = isinstance(left, lattice_planes.PlaneReading)
    right_valid = isinstance(right, lattice_planes.PlaneReading)
    shift_support = _shift_support(match, left if left_valid else None,
                                   right if right_valid else None)

    top_refusal = None
    if not left_valid or not right_valid:
        top_refusal = "reading_missing"
    elif left_b is None or right_b is None:
        top_refusal = "invalid_period_fraction"
    elif (left.frame != match.left or right.frame != match.right
          or left.axis != match.left_axis or right.axis != match.right_axis):
        top_refusal = "reading_identity_mismatch"

    left_alt = right_alt = None
    if top_refusal is None:
        left_matches = left.matching_alternatives(left_b)
        right_matches = right.matching_alternatives(right_b)
        if len(left_matches) != 1 or len(right_matches) != 1:
            top_refusal = "alternative_not_retained"
        else:
            left_alt, right_alt = left_matches[0], right_matches[0]
            if (_rational_b(float(left_alt.blocks_per_period)) is None
                    or _rational_b(float(right_alt.blocks_per_period)) is None):
                top_refusal = "invalid_period_fraction"

    phase_records: tuple[PeriodicResidueCandidates, ...] = ()
    if top_refusal is None:
        phase_records = tuple(
            _candidate_record(reading, plane_index, b_value, alternative, local_axis)
            for side, reading, plane_index, b_value, alternative in (
                ("left", left, int(match.left_plane),
                 float(left_alt.blocks_per_period), left_alt),
                ("right", right, int(match.right_plane),
                 float(right_alt.blocks_per_period), right_alt),
            )
            for local_axis in (0, 1)
        )

    return HypothesisPhaseShiftEvidence(
        left_frame=str(match.left), right_frame=str(match.right),
        left_plane=int(match.left_plane), right_plane=int(match.right_plane),
        left_axis=int(match.left_axis), right_axis=int(match.right_axis),
        left_b=left_b, right_b=right_b,
        quadrant_of_right=int(match.quadrant_of_right),
        phase_candidates=phase_records, relative_shift=shift_support,
        refusal=top_refusal)
