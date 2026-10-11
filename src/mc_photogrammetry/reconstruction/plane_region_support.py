"""M4 V1 isolated phase-aligned region support (provenance preserving).

This module implements only the isolated V1 contract of
``docs/reports/M4-placement-volume-bridge-contract-2026-09-19.md``: project the
*original retained region mask*, intersected with the caller's validated usable
mask, onto unit cells of a phase-aligned local block grid and classify each cell
as ``supported`` or ``unknown`` under a conservative full-area policy.

It defines no face placement, no volume emission, no block states or material
inference, no CLI/UI, no persistence schema, no litematic wiring, no
manual-coordinate input and no promotion gate.  It never imports the accepted
V2-B solver, the placement, volume, export or litematic modules.

Load-bearing rules:

* the cell grid is the retained alternative's own block lattice: a unit cell
  ``(row, column)`` covers block coordinates
  ``[u0 + column, u0 + column + 1] x [v0 + row, v0 + row + 1]`` with an exact
  rational ``origin_uv`` that may be fractional (phase-aligned);
* the cell's projected quad is obtained from an exact rational homography built
  from the declared calibration inputs at the retained alternative's exact
  ``distance_blocks`` -- never from a proposal bounding polygon, never by
  shifting window-relative pixel counts, and never from the rounded ``float64``
  coefficients of :func:`lattice_planes.rectification_homography` (which is
  called only to detect an overflowing projection);
* a cell is ``supported`` only when its *entire* projected area is established
  inside the region mask, inside the usable mask and inside the image bounds,
  all under exact rational arithmetic.  Every other cell is ``unknown`` with
  its region/usable/frame excluded areas retained as diagnostic evidence.  Air
  is never inferred from an absent mask pixel;
* the supplied V2-A ``ReadingIdentity`` is rebound to the reading and its
  canonical universe, and the sealed record cross-links it to the calibration,
  retained alternative, region-mask digest and image digest, so none of them can
  be swapped while keeping the old identity.

All arithmetic is exact: IEEE-754 inputs are dyadic rationals, converted with
``Fraction(float)``; the homography composition, polygon clipping and area
accumulation stay rational, so a cell is supported only on an exact zero
excluded area.  Every input and projection precondition is validated before any
budget decision, so a degenerate or overflowing homography never reports
``budget_exhausted``.  Work is bounded by an explicit
:class:`AlignedSupportBudget`; exceeding any cap discards every partial cell and
returns an atomic ``budget_exhausted`` refusal.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Mapping, Sequence

import numpy as np

from mc_photogrammetry.reconstruction import lattice_planes
from mc_photogrammetry.reconstruction import plane_endpoint_provenance as v2a


__all__ = [
    "V1_SUPPORT_SCHEMA",
    "V1_SUPPORT_PRODUCER_VERSION",
    "V1_SUPPORT_COVERAGE_POLICY",
    "V1_SUPPORT_CELL_STATES",
    "V1_SUPPORT_CELL_EVIDENCE",
    "V1_SUPPORT_REFUSAL_REASONS",
    "AlignedSupportBudget",
    "AlignedSupportWork",
    "AlignedSupportRefusal",
    "RegionSupportCalibration",
    "RetainedAlternative",
    "AlignedSupportCell",
    "AlignedRegionSupport",
    "make_support_budget",
    "usable_mask_sha256",
    "canonical_alternative_order",
    "support_sort_key",
    "canonical_support_order",
    "build_aligned_region_support",
]


V1_SUPPORT_SCHEMA = "m4-v1-aligned-region-support-1"
V1_SUPPORT_PRODUCER_VERSION = "v1-region-support-producer-1"
V1_SUPPORT_CALIBRATION_VERSION = "v1-region-support-calibration-1"
V1_SUPPORT_COVERAGE_POLICY = "conservative_full_area_v1"
V1_SUPPORT_CELL_STATES: tuple[str, ...] = ("supported", "unknown")
V1_SUPPORT_CELL_EVIDENCE: tuple[str, ...] = (
    "full", "outside_frame", "outside_region", "outside_usable", "degenerate",
)
V1_SUPPORT_REFUSAL_REASONS: tuple[str, ...] = (
    "invalid_input",
    "missing_provenance",
    "provenance_mismatch",
    "invalid_region_mask",
    "invalid_usable_mask",
    "image_shape_mismatch",
    "calibration_mismatch",
    "alternative_not_retained",
    "invalid_origin",
    "invalid_grid",
    "budget_exhausted",
)

_ROW_CONVENTION = "v"
_COLUMN_CONVENTION = "u"


# ---------------------------------------------------------------------------
# Canonical JSON, digests, exact-rational parsing
# ---------------------------------------------------------------------------

def _canonical_json(value: Any) -> str:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _json_value(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON values must be finite")
        return float(value)
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    raise ValueError("value is not a deterministic JSON primitive")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _hex64(value: Any) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _finite(value: Any) -> float | None:
    if type(value) not in (int, float) or type(value) is bool:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _fraction(value: Any) -> Fraction:
    """Parse a caller rational: exact int, exact float, or ``(num, den)`` pair."""
    if isinstance(value, Fraction):
        return value
    if type(value) is int and type(value) is not bool:
        return Fraction(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("rational must be finite")
        return Fraction(value)
    if (isinstance(value, (tuple, list)) and len(value) == 2
            and all(type(part) is int for part in value)):
        numerator, denominator = value
        if denominator <= 0:
            raise ValueError("rational denominator must be positive")
        return Fraction(numerator, denominator)
    raise ValueError("rational must be an int, float or (numerator, denominator)")


def _pair(value: Fraction) -> tuple[int, int]:
    return (int(value.numerator), int(value.denominator))


def _validate_signed_pair(value: Any) -> bool:
    return (isinstance(value, tuple) and len(value) == 2
            and all(type(part) is int for part in value)
            and value[1] > 0 and math.gcd(value[0], value[1]) == 1)


def _validate_nonneg_pair(value: Any) -> bool:
    return (_validate_signed_pair(value) and value[0] >= 0)


# ---------------------------------------------------------------------------
# Budget, work and refusal
# ---------------------------------------------------------------------------

_BUDGET_FIELDS: tuple[str, ...] = (
    "max_cells",
    "max_region_pixel_tests",
    "max_usable_pixel_tests",
    "max_polygon_vertices",
    "max_supported_cells",
)
_WORK_FIELDS: tuple[str, ...] = (
    "cells_examined",
    "frame_clips",
    "region_pixel_tests",
    "usable_pixel_tests",
    "polygon_vertices",
    "supported_cells",
    "unknown_cells",
)
_WORK_TO_CAP = {
    "cells_examined": "max_cells",
    "region_pixel_tests": "max_region_pixel_tests",
    "usable_pixel_tests": "max_usable_pixel_tests",
    "polygon_vertices": "max_polygon_vertices",
    "supported_cells": "max_supported_cells",
}
_BUDGET_CAP_WORK = {cap: work for work, cap in _WORK_TO_CAP.items()}


def _work_counts_consistent(work: "AlignedSupportWork") -> bool:
    """Semantic work relations that a freshly resealed record could otherwise forge.

    Every examined cell contributes at most one state and at most one frame clip,
    so ``supported_cells + unknown_cells`` and ``frame_clips`` can never exceed
    ``cells_examined``.  The cap relations are enforced where a cap is known: the
    refusal's ``observed``/cap binding and the support record's grid/state
    cross-checks.
    """
    return (work.supported_cells + work.unknown_cells <= work.cells_examined
            and work.frame_clips <= work.cells_examined)


@dataclass(frozen=True, slots=True)
class AlignedSupportBudget:
    """Explicit positive whole-request limits.  No limit has a default."""

    max_cells: int
    max_region_pixel_tests: int
    max_usable_pixel_tests: int
    max_polygon_vertices: int
    max_supported_cells: int
    budget_sha256: str

    def __post_init__(self) -> None:
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 1
               for name in _BUDGET_FIELDS):
            raise ValueError("support budget limits must be positive exact integers")
        if (not _hex64(self.budget_sha256)
                or self.budget_sha256 != _sha256(self._payload())):
            raise ValueError("support budget digest mismatch")

    def _payload(self) -> dict:
        return {name: getattr(self, name) for name in _BUDGET_FIELDS}

    def to_record(self) -> dict:
        record = self._payload()
        record["budget_sha256"] = self.budget_sha256
        return record


def make_support_budget(**caps: int) -> AlignedSupportBudget:
    """Build a self-validating budget from explicit caps (no defaults)."""
    payload: dict[str, int] = {}
    for name in _BUDGET_FIELDS:
        if name not in caps:
            raise ValueError("support budget is missing a limit")
        value = caps[name]
        if type(value) is not int or value < 1:
            raise ValueError("support budget limits must be positive exact integers")
        payload[name] = value
    return AlignedSupportBudget(**payload, budget_sha256=_sha256(payload))


@dataclass(frozen=True, slots=True)
class AlignedSupportWork:
    """Exact observed work counters for one support request."""

    cells_examined: int
    frame_clips: int
    region_pixel_tests: int
    usable_pixel_tests: int
    polygon_vertices: int
    supported_cells: int
    unknown_cells: int
    work_sha256: str

    def __post_init__(self) -> None:
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 0
               for name in _WORK_FIELDS):
            raise ValueError("support work counters must be non-negative integers")
        if not _work_counts_consistent(self):
            raise ValueError(
                "support work state counts exceed the examined cell count")
        if (not _hex64(self.work_sha256)
                or self.work_sha256 != _sha256(self._payload())):
            raise ValueError("support work digest mismatch")

    def _payload(self) -> dict:
        return {name: getattr(self, name) for name in _WORK_FIELDS}

    def to_record(self) -> dict:
        record = self._payload()
        record["work_sha256"] = self.work_sha256
        return record


def _make_work(counters: Mapping[str, int]) -> AlignedSupportWork:
    payload: dict[str, int] = {}
    for name in _WORK_FIELDS:
        value = counters.get(name, 0)
        if type(value) is not int or value < 0:
            raise ValueError("support work counters must be non-negative integers")
        payload[name] = value
    return AlignedSupportWork(**payload, work_sha256=_sha256(payload))


@dataclass(frozen=True, slots=True)
class AlignedSupportRefusal:
    """One typed, atomic, canonical support refusal.  It carries no cells."""

    reason: str
    detail: dict
    detail_sha256: str
    work: AlignedSupportWork | None
    refusal_sha256: str

    def __post_init__(self) -> None:
        if self.reason not in V1_SUPPORT_REFUSAL_REASONS:
            raise ValueError("support refusal reason is not in the closed enum")
        normalized = _json_value(self.detail)
        if not isinstance(normalized, dict):
            raise ValueError("support refusal detail must be a JSON object")
        encoded = _canonical_json(normalized)
        if self.detail_sha256 != hashlib.sha256(encoded.encode("utf-8")).hexdigest():
            raise ValueError("support refusal detail digest mismatch")
        if self.work is not None and not isinstance(self.work, AlignedSupportWork):
            raise ValueError("support refusal work is malformed")
        # Reason-specific invariants.  A freshly resealed refusal recomputes its
        # own digests, so only these semantic checks can reject a contradiction.
        if "cells" in normalized:
            raise ValueError("support refusal must not embed partial cells")
        for name in ("limit", "observed"):
            if name in normalized:
                value = normalized[name]
                if type(value) is not int or value <= 0:
                    raise ValueError(
                        "support refusal %s must be a positive exact integer" % name)
        if self.reason == "budget_exhausted":
            if self.work is None:
                raise ValueError("budget refusal requires observed work evidence")
            if set(normalized) != {"cap", "limit", "observed"}:
                raise ValueError("budget refusal detail is malformed")
            cap = normalized["cap"]
            if cap not in _BUDGET_CAP_WORK:
                raise ValueError("budget refusal cap is not a support cap")
            observed = normalized["observed"]
            limit = normalized["limit"]
            if observed <= limit:
                raise ValueError("budget refusal observed must exceed its limit")
            if not _work_counts_consistent(self.work):
                raise ValueError("budget refusal work counters are inconsistent")
            # ``max_cells`` is the only ex-ante cap: it is decided before any
            # cell is examined, so a planned refusal must carry all-zero work.  A
            # freshly resealed refusal recomputes its own digests, so only this
            # semantic check can reject impossible planned counters.
            if cap == "max_cells" or self.work.cells_examined == 0:
                if cap != "max_cells":
                    raise ValueError(
                        "budget refusal without examined cells must be a planned "
                        "max_cells stop")
                if any(getattr(self.work, name) != 0 for name in _WORK_FIELDS):
                    raise ValueError(
                        "planned budget refusal must have every work counter "
                        "zero")
            elif observed != getattr(self.work, _BUDGET_CAP_WORK[cap]):
                raise ValueError(
                    "budget refusal observed disagrees with its work counter")
        elif self.work is not None:
            raise ValueError("non-budget refusal must not carry work evidence")
        payload = {
            "reason": self.reason,
            "detail_sha256": self.detail_sha256,
            "work": None if self.work is None else self.work.to_record(),
        }
        if (not _hex64(self.refusal_sha256)
                or self.refusal_sha256 != _sha256(payload)):
            raise ValueError("support refusal digest mismatch")
        object.__setattr__(self, "detail", normalized)

    def to_record(self) -> dict:
        return {
            "reason": self.reason,
            "detail": dict(self.detail),
            "detail_sha256": self.detail_sha256,
            "work": None if self.work is None else self.work.to_record(),
            "refusal_sha256": self.refusal_sha256,
        }


def _make_refusal(reason: str, detail: Any,
                  work: AlignedSupportWork | None = None) -> AlignedSupportRefusal:
    if reason not in V1_SUPPORT_REFUSAL_REASONS:
        reason = "invalid_input"
    normalized = _json_value({} if detail is None else detail)
    detail_sha = hashlib.sha256(
        _canonical_json(normalized).encode("utf-8")).hexdigest()
    payload = {
        "reason": reason,
        "detail_sha256": detail_sha,
        "work": None if work is None else work.to_record(),
    }
    return AlignedSupportRefusal(reason, normalized, detail_sha, work,
                                 _sha256(payload))


class _InputRefusal(Exception):
    """Internal typed input refusal; the public builder maps it to a record."""

    def __init__(self, reason: str, detail: Any):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


class _BudgetStop(Exception):
    """Internal exact cap excess; every partial artifact is discarded."""

    def __init__(self, cap: str, limit: int, observed: int):
        super().__init__(cap)
        self.cap = cap
        self.limit = limit
        self.observed = observed


class _SupportCounter:
    __slots__ = ("budget", "cells_examined", "frame_clips",
                 "region_pixel_tests", "usable_pixel_tests",
                 "polygon_vertices", "supported_cells", "unknown_cells")

    def __init__(self, budget: AlignedSupportBudget):
        self.budget = budget
        self.cells_examined = 0
        self.frame_clips = 0
        self.region_pixel_tests = 0
        self.usable_pixel_tests = 0
        self.polygon_vertices = 0
        self.supported_cells = 0
        self.unknown_cells = 0

    def check(self) -> None:
        for work_name in _WORK_FIELDS:
            cap_name = _WORK_TO_CAP.get(work_name)
            if cap_name is None:
                continue
            observed = getattr(self, work_name)
            limit = getattr(self.budget, cap_name)
            if observed > limit:
                raise _BudgetStop(cap_name, limit, observed)

    def snapshot(self) -> AlignedSupportWork:
        return _make_work({name: getattr(self, name) for name in _WORK_FIELDS})


# ---------------------------------------------------------------------------
# Bound provenance records
# ---------------------------------------------------------------------------

def _calibration_payload(value: "RegionSupportCalibration") -> dict:
    return {
        "calibration_version": value.calibration_version,
        "axis_matrix": [list(row) for row in value.axis_matrix],
        "focal": value.focal,
        "principal_point": list(value.principal_point),
        "centre": list(value.centre),
        "side": value.side,
    }


@dataclass(frozen=True, slots=True)
class RegionSupportCalibration:
    """Frozen rectification calibration bound to one support request."""

    calibration_version: str
    axis_matrix: tuple[tuple[float, float, float], ...]
    focal: float
    principal_point: tuple[float, float]
    centre: tuple[float, float]
    side: float
    calibration_sha256: str

    def __post_init__(self) -> None:
        if self.calibration_version != V1_SUPPORT_CALIBRATION_VERSION:
            raise ValueError("support calibration version is unsupported")
        if (not isinstance(self.axis_matrix, tuple) or len(self.axis_matrix) != 3
                or any(not isinstance(row, tuple) or len(row) != 3
                       for row in self.axis_matrix)):
            raise ValueError("support calibration axis matrix is malformed")
        if any(_finite(value) is None for row in self.axis_matrix
               for value in row):
            raise ValueError("support calibration axis matrix is not finite")
        if _finite(self.focal) is None or self.focal <= 0.0:
            raise ValueError("support calibration focal is invalid")
        for name in ("principal_point", "centre"):
            value = getattr(self, name)
            if (not isinstance(value, tuple) or len(value) != 2
                    or any(_finite(part) is None for part in value)):
                raise ValueError("support calibration pair is malformed")
        if _finite(self.side) is None or self.side == 0.0:
            raise ValueError("support calibration side is invalid")
        if (not _hex64(self.calibration_sha256)
                or self.calibration_sha256 != _sha256(_calibration_payload(self))):
            raise ValueError("support calibration digest mismatch")

    @classmethod
    def build(cls, *, axis_matrix: Sequence[Sequence[float]], focal: float,
              principal_point: Sequence[float], centre: Sequence[float],
              side: float) -> "RegionSupportCalibration":
        provisional = object.__new__(cls)
        fields = {
            "calibration_version": V1_SUPPORT_CALIBRATION_VERSION,
            "axis_matrix": tuple(tuple(float(value) for value in row)
                                 for row in axis_matrix),
            "focal": float(focal),
            "principal_point": (float(principal_point[0]), float(principal_point[1])),
            "centre": (float(centre[0]), float(centre[1])),
            "side": float(side),
        }
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "calibration_sha256", "")
        object.__setattr__(provisional, "calibration_sha256",
                           _sha256(_calibration_payload(provisional)))
        cls.__post_init__(provisional)
        return provisional

    def to_record(self) -> dict:
        record = _calibration_payload(self)
        record["calibration_sha256"] = self.calibration_sha256
        return record


def _alternative_payload(value: "RetainedAlternative") -> dict:
    return {
        "blocks_per_period": value.blocks_per_period,
        "distance_blocks": value.distance_blocks,
        "support": value.support,
    }


@dataclass(frozen=True, slots=True)
class RetainedAlternative:
    """The one retained distance/scale alternative a support is built on."""

    blocks_per_period: float
    distance_blocks: float
    support: float
    alternative_sha256: str

    def __post_init__(self) -> None:
        if _finite(self.blocks_per_period) is None or self.blocks_per_period <= 0.0:
            raise ValueError("retained alternative period is invalid")
        if _finite(self.distance_blocks) is None or self.distance_blocks <= 0.0:
            raise ValueError("retained alternative distance is invalid")
        if _finite(self.support) is None or self.support < 0.0:
            raise ValueError("retained alternative support is invalid")
        if (not _hex64(self.alternative_sha256)
                or self.alternative_sha256 != _sha256(_alternative_payload(self))):
            raise ValueError("retained alternative digest mismatch")

    @classmethod
    def build(cls, alternative: "lattice_planes.DistanceAlternative"
              ) -> "RetainedAlternative":
        if not isinstance(alternative, lattice_planes.DistanceAlternative):
            raise ValueError("retained alternative has an invalid type")
        payload = {
            "blocks_per_period": float(alternative.blocks_per_period),
            "distance_blocks": float(alternative.distance_blocks),
            "support": float(alternative.support),
        }
        return cls(**payload, alternative_sha256=_sha256(payload))

    def to_record(self) -> dict:
        record = _alternative_payload(self)
        record["alternative_sha256"] = self.alternative_sha256
        return record


def canonical_alternative_order(
    alternatives: Sequence["lattice_planes.DistanceAlternative"],
) -> tuple["lattice_planes.DistanceAlternative", ...]:
    """Deterministic content order for a reading's retained alternatives.

    The order is ``(blocks_per_period, distance_blocks, support)`` and does not
    depend on the caller's insertion order or on object identity.
    """
    if not isinstance(alternatives, Sequence):
        raise ValueError("alternatives must be a sequence")
    for alternative in alternatives:
        if not isinstance(alternative, lattice_planes.DistanceAlternative):
            raise ValueError("alternatives contain a foreign record")
    return tuple(sorted(alternatives, key=lambda item: (
        float(item.blocks_per_period), float(item.distance_blocks),
        float(item.support))))


def _canonical_reading(
        reading: "lattice_planes.PlaneReading") -> "lattice_planes.PlaneReading":
    """Return the reading with alternatives in canonical content order.

    Reordered alternatives are the same reading under the V1 canonical policy,
    so identity/content checks operate on this canonical form and the support
    digest stays independent of insertion order.
    """
    ordered = canonical_alternative_order(reading.alternatives)
    if ordered == tuple(reading.alternatives):
        return reading
    return replace(reading, alternatives=ordered)


def _validate_identity_content(
        reading: "lattice_planes.PlaneReading",
        identity: v2a.ReadingIdentity,
        universe: Sequence["lattice_planes.PlaneReading"]) -> None:
    """Bind a supplied V2-A identity to the reading and its canonical universe.

    The identity's own digest is only as strong as the content it was built
    from.  A changed alternative/distance, axis, frame, source region, region
    mask, rectification/calibration or reading-order content must not validate
    against the old identity, so the identity is recomputed from the canonical
    reading and the canonical universe and compared field for field; the
    universe entry itself must then content-equal the reading (which covers the
    rectification that V2-A's order key omits).
    """
    if (not isinstance(universe, Sequence) or isinstance(universe, (str, bytes))
            or len(universe) == 0):
        raise _InputRefusal("missing_provenance", {"field": "reading_universe"})
    for candidate in universe:
        if not isinstance(candidate, lattice_planes.PlaneReading):
            raise _InputRefusal("missing_provenance",
                                {"field": "reading_universe"})
    try:
        canonical = _canonical_reading(reading)
    except (ValueError, TypeError, AttributeError) as error:
        raise _InputRefusal("provenance_mismatch", {
            "detail": "reading alternatives are not canonicalizable"}) from error
    evidence = canonical.region_evidence
    source_region_id = None if evidence is None else evidence.source_region_id
    if (identity.frame != canonical.frame
            or identity.reading_axis != canonical.axis
            or identity.source_region_id != source_region_id):
        raise _InputRefusal("provenance_mismatch", {
            "frame": canonical.frame, "axis": int(canonical.axis)})
    if identity.reading_order_key_sha256 != v2a._reading_order_key_sha256(
            canonical):
        raise _InputRefusal("provenance_mismatch",
                            {"field": "reading_order_key_sha256"})
    try:
        canonical_universe = tuple(_canonical_reading(candidate)
                                   for candidate in universe)
        ordered = v2a._ordered_readings(canonical_universe)
        universe_sha = v2a._canonical_universe_sha256(canonical_universe)
        lookup = v2a._canonical_index_lookup(ordered)
        expected = v2a._reading_identity(canonical, lookup, universe_sha)
    except v2a.V2AFailure as error:
        raise _InputRefusal("provenance_mismatch",
                            {"detail": str(error)[:160]}) from error
    except (ValueError, TypeError, AttributeError, KeyError) as error:
        raise _InputRefusal("provenance_mismatch", {
            "detail": "reading universe is not canonicalizable"}) from error
    if expected != identity:
        raise _InputRefusal("provenance_mismatch", {
            "detail": "reading identity is not its canonical universe identity"})
    frame_values = ordered.get(canonical.frame, ())
    if (identity.canonical_index >= len(frame_values)
            or frame_values[identity.canonical_index] != canonical):
        raise _InputRefusal("provenance_mismatch", {
            "detail": "reading content differs from its canonical universe entry"})


# ---------------------------------------------------------------------------
# Cell and support records
# ---------------------------------------------------------------------------

def _classify_cell(cell_area: Fraction, frame_excluded: Fraction,
                   region_excluded: Fraction,
                   usable_excluded: Fraction) -> str:
    if cell_area == 0:
        return "degenerate"
    if frame_excluded > 0:
        return "outside_frame"
    if region_excluded > 0:
        return "outside_region"
    if usable_excluded > 0:
        return "outside_usable"
    return "full"


def _cell_payload(value: "AlignedSupportCell") -> dict:
    return {
        "row": value.row,
        "column": value.column,
        "state": value.state,
        "evidence": value.evidence,
        "cell_area": list(value.cell_area),
        "region_excluded": list(value.region_excluded),
        "usable_excluded": list(value.usable_excluded),
        "frame_excluded": list(value.frame_excluded),
    }


@dataclass(frozen=True, slots=True)
class AlignedSupportCell:
    """One unit cell's typed state and exact excluded-area diagnostic evidence."""

    row: int
    column: int
    state: str
    evidence: str
    cell_area: tuple[int, int]
    region_excluded: tuple[int, int]
    usable_excluded: tuple[int, int]
    frame_excluded: tuple[int, int]
    cell_sha256: str

    def __post_init__(self) -> None:
        if (type(self.row) is not int or self.row < 0
                or type(self.column) is not int or self.column < 0):
            raise ValueError("support cell indices are invalid")
        if self.state not in V1_SUPPORT_CELL_STATES:
            raise ValueError("support cell state is not in the closed enum")
        if self.evidence not in V1_SUPPORT_CELL_EVIDENCE:
            raise ValueError("support cell evidence is not in the closed enum")
        for name in ("cell_area", "region_excluded", "usable_excluded",
                     "frame_excluded"):
            if not _validate_nonneg_pair(getattr(self, name)):
                raise ValueError("support cell areas must be reduced non-negative rationals")
        cell_area = Fraction(*self.cell_area)
        frame_excluded = Fraction(*self.frame_excluded)
        region_excluded = Fraction(*self.region_excluded)
        usable_excluded = Fraction(*self.usable_excluded)
        if not (frame_excluded <= region_excluded <= cell_area
                and frame_excluded <= usable_excluded <= cell_area):
            raise ValueError("support cell area evidence is inconsistent")
        expected_evidence = _classify_cell(
            cell_area, frame_excluded, region_excluded, usable_excluded)
        if self.evidence != expected_evidence:
            raise ValueError("support cell evidence disagrees with its areas")
        expected_state = ("supported" if expected_evidence == "full"
                          else "unknown")
        if self.state != expected_state:
            raise ValueError("support cell state disagrees with its evidence")
        if (not _hex64(self.cell_sha256)
                or self.cell_sha256 != _sha256(_cell_payload(self))):
            raise ValueError("support cell digest mismatch")

    @classmethod
    def build(cls, *, row: int, column: int, cell_area: Fraction,
              region_excluded: Fraction, usable_excluded: Fraction,
              frame_excluded: Fraction) -> "AlignedSupportCell":
        evidence = _classify_cell(cell_area, frame_excluded, region_excluded,
                                  usable_excluded)
        state = "supported" if evidence == "full" else "unknown"
        payload = {
            "row": row,
            "column": column,
            "state": state,
            "evidence": evidence,
            "cell_area": list(_pair(cell_area)),
            "region_excluded": list(_pair(region_excluded)),
            "usable_excluded": list(_pair(usable_excluded)),
            "frame_excluded": list(_pair(frame_excluded)),
        }
        return cls(row=row, column=column, state=state, evidence=evidence,
                   cell_area=_pair(cell_area),
                   region_excluded=_pair(region_excluded),
                   usable_excluded=_pair(usable_excluded),
                   frame_excluded=_pair(frame_excluded),
                   cell_sha256=_sha256(payload))

    def to_record(self) -> dict:
        record = _cell_payload(self)
        record["cell_sha256"] = self.cell_sha256
        return record


def _reading_binding_payload(value: "AlignedRegionSupport") -> dict:
    """Cross-link that binds the supplied reading identity to real content.

    The V2-A ``ReadingIdentity`` alone does not cover the reading's
    rectification/calibration, and its order key does not survive a freshly
    resealed record.  This payload ties the identity to the calibration,
    retained alternative, source-region mask digest and image digest that the
    support was actually built from, so none of them can be swapped while
    keeping the old identity binding.
    """
    return {
        "reading_identity": value.reading_identity.to_record(),
        "reading_identity_sha256": value.reading_identity_sha256,
        "source_region_id": value.source_region_id,
        "region_mask_sha256": value.region_mask_sha256,
        "image_sha256": value.image_sha256,
        "image_shape": list(value.image_shape),
        "calibration": value.calibration.to_record(),
        "alternative": value.alternative.to_record(),
    }


def _support_payload(value: "AlignedRegionSupport") -> dict:
    return {
        "schema_version": value.schema_version,
        "producer_version": value.producer_version,
        "coverage_policy": value.coverage_policy,
        "source_region_id": value.source_region_id,
        "region_mask_sha256": value.region_mask_sha256,
        "image_sha256": value.image_sha256,
        "image_shape": list(value.image_shape),
        "usable_mask_sha256": value.usable_mask_sha256,
        "reading_identity": value.reading_identity.to_record(),
        "reading_identity_sha256": value.reading_identity_sha256,
        "reading_binding_sha256": value.reading_binding_sha256,
        "calibration": value.calibration.to_record(),
        "alternative": value.alternative.to_record(),
        "origin_uv": [list(value.origin_uv[0]), list(value.origin_uv[1])],
        "grid_shape": list(value.grid_shape),
        "row_label": value.row_label,
        "column_label": value.column_label,
        "row_convention": value.row_convention,
        "column_convention": value.column_convention,
        "phase_anchor_sha256": value.phase_anchor_sha256,
        "cells": [cell.to_record() for cell in value.cells],
        "work": value.work.to_record(),
    }


@dataclass(frozen=True, slots=True)
class AlignedRegionSupport:
    """The versioned, immutable, provenance-bound phase-aligned support map."""

    schema_version: str
    producer_version: str
    coverage_policy: str
    source_region_id: str
    region_mask_sha256: str
    image_sha256: str
    image_shape: tuple[int, int]
    usable_mask_sha256: str
    reading_identity: v2a.ReadingIdentity
    reading_identity_sha256: str
    reading_binding_sha256: str
    calibration: RegionSupportCalibration
    alternative: RetainedAlternative
    origin_uv: tuple[tuple[int, int], tuple[int, int]]
    grid_shape: tuple[int, int]
    row_label: int
    column_label: int
    row_convention: str
    column_convention: str
    phase_anchor_sha256: str | None
    cells: tuple[AlignedSupportCell, ...]
    work: AlignedSupportWork
    support_sha256: str

    def __post_init__(self) -> None:
        if (self.schema_version != V1_SUPPORT_SCHEMA
                or self.producer_version != V1_SUPPORT_PRODUCER_VERSION
                or self.coverage_policy != V1_SUPPORT_COVERAGE_POLICY):
            raise ValueError("support record version or policy is unsupported")
        if (not _hex64(self.source_region_id)
                or not _hex64(self.region_mask_sha256)
                or not _hex64(self.image_sha256)
                or not _hex64(self.usable_mask_sha256)):
            raise ValueError("support record digests are malformed")
        if (not isinstance(self.image_shape, tuple) or len(self.image_shape) != 2
                or any(type(value) is not int or value <= 0
                       for value in self.image_shape)):
            raise ValueError("support image shape is invalid")
        if (not isinstance(self.reading_identity, v2a.ReadingIdentity)
                or self.reading_identity_sha256
                != self.reading_identity.identity_sha256
                or self.reading_identity.source_region_id
                != self.source_region_id):
            raise ValueError("support reading identity binding is invalid")
        if not isinstance(self.calibration, RegionSupportCalibration):
            raise ValueError("support calibration is malformed")
        if not isinstance(self.alternative, RetainedAlternative):
            raise ValueError("support alternative is malformed")
        for name in ("origin_uv",):
            value = getattr(self, name)
            if (not isinstance(value, tuple) or len(value) != 2
                    or any(not _validate_signed_pair(pair) for pair in value)):
                raise ValueError("support grid origin is not an exact rational pair")
        if (not isinstance(self.grid_shape, tuple) or len(self.grid_shape) != 2
                or any(type(value) is not int or value <= 0
                       for value in self.grid_shape)):
            raise ValueError("support grid shape is invalid")
        if (type(self.row_label) is not int or type(self.column_label) is not int
                or (self.column_label, self.row_label)
                != tuple(label for label in range(3)
                         if label != self.reading_identity.reading_axis)
                or self.row_convention != _ROW_CONVENTION
                or self.column_convention != _COLUMN_CONVENTION):
            raise ValueError("support row/column convention is invalid")
        if self.phase_anchor_sha256 is not None \
                and not _hex64(self.phase_anchor_sha256):
            raise ValueError("support phase anchor digest is malformed")
        rows, columns = self.grid_shape
        if not isinstance(self.cells, tuple) or len(self.cells) != rows * columns:
            raise ValueError("support cell count does not match its grid")
        expected = tuple((row, column) for row in range(rows)
                         for column in range(columns))
        if any(not isinstance(cell, AlignedSupportCell) for cell in self.cells):
            raise ValueError("support cells are malformed")
        if tuple((cell.row, cell.column) for cell in self.cells) != expected:
            raise ValueError("support cells are not in canonical row-major order")
        if not isinstance(self.work, AlignedSupportWork):
            raise ValueError("support work record is malformed")
        if not _work_counts_consistent(self.work):
            raise ValueError("support work counters are inconsistent")
        if self.work.cells_examined != rows * columns:
            raise ValueError("support work does not cover every cell")
        supported = sum(1 for cell in self.cells if cell.state == "supported")
        if (self.work.supported_cells != supported
                or self.work.unknown_cells != rows * columns - supported):
            raise ValueError("support work state counts disagree with its cells")
        if (not _hex64(self.reading_binding_sha256)
                or self.reading_binding_sha256
                != _sha256(_reading_binding_payload(self))):
            raise ValueError("support reading binding digest mismatch")
        if (not _hex64(self.support_sha256)
                or self.support_sha256 != _sha256(_support_payload(self))):
            raise ValueError("support digest mismatch")

    @classmethod
    def build(cls, **fields: Any) -> "AlignedRegionSupport":
        provisional = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "reading_binding_sha256",
                           _sha256(_reading_binding_payload(provisional)))
        object.__setattr__(provisional, "support_sha256", "")
        object.__setattr__(provisional, "support_sha256",
                           _sha256(_support_payload(provisional)))
        cls.__post_init__(provisional)
        return provisional

    def to_record(self) -> dict:
        record = _support_payload(self)
        record["support_sha256"] = self.support_sha256
        return record

    @property
    def cell_by_index(self) -> tuple[AlignedSupportCell, ...]:
        return self.cells

    @property
    def supported_count(self) -> int:
        return sum(1 for cell in self.cells if cell.state == "supported")

    @property
    def unknown_count(self) -> int:
        return sum(1 for cell in self.cells if cell.state == "unknown")


def support_sort_key(record: Any) -> tuple:
    """Deterministic content key for a support or refusal record."""
    if isinstance(record, AlignedRegionSupport):
        return ("support", record.source_region_id, record.image_sha256,
                record.reading_identity.frame,
                record.reading_identity.canonical_index,
                record.origin_uv, record.grid_shape, record.support_sha256)
    if isinstance(record, AlignedSupportRefusal):
        return ("refusal", record.reason, record.detail_sha256,
                record.refusal_sha256)
    raise ValueError("support sort key requires a support or refusal record")


def canonical_support_order(records: Sequence[Any]) -> tuple:
    """Canonicalize a reordered input sequence of support/refusal records."""
    if not isinstance(records, Sequence):
        raise ValueError("support records must be a sequence")
    return tuple(sorted(records, key=support_sort_key))


# ---------------------------------------------------------------------------
# Usable-mask digest
# ---------------------------------------------------------------------------

_USABLE_MASK_HEADER = {
    "schema": "v1-usable-mask-1",
    "encoding": "packbits-base64",
    "bit_order": "little",
}


def usable_mask_sha256(mask: np.ndarray) -> str:
    """Content digest of a validated 2D boolean usable mask."""
    if (not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.size == 0
            or mask.dtype != np.dtype(bool)):
        raise ValueError("usable mask must be a non-empty 2D boolean array")
    binary = np.ascontiguousarray(mask)
    packed = np.packbits(binary.reshape(-1), bitorder="little").tobytes()
    header = dict(_USABLE_MASK_HEADER, image_shape=[int(binary.shape[0]),
                                                    int(binary.shape[1])])
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded + b"\0" + packed).hexdigest()


# ---------------------------------------------------------------------------
# Exact polygon geometry
# ---------------------------------------------------------------------------

def _clip_line(points: tuple, index: int, bound: Fraction,
               keep_greater: bool) -> tuple:
    if not points:
        return ()
    result: list = []
    count = len(points)
    for position in range(count):
        current = points[position]
        previous = points[position - 1]
        current_inside = (current[index] >= bound if keep_greater
                          else current[index] <= bound)
        previous_inside = (previous[index] >= bound if keep_greater
                           else previous[index] <= bound)
        if current_inside != previous_inside:
            denominator = current[index] - previous[index]
            if denominator != 0:
                ratio = (bound - previous[index]) / denominator
                result.append((previous[0] + ratio * (current[0] - previous[0]),
                               previous[1] + ratio * (current[1] - previous[1])))
        if current_inside:
            result.append(current)
    return _dedupe(result)


def _dedupe(points: Sequence) -> tuple:
    result: list = []
    for point in points:
        if not result or result[-1] != point:
            result.append(point)
    if len(result) > 1 and result[0] == result[-1]:
        result.pop()
    return tuple(result)


def _clip_rect(points: Sequence, xmin: Fraction, xmax: Fraction,
               ymin: Fraction, ymax: Fraction) -> tuple:
    result = _dedupe(points)
    result = _clip_line(result, 0, xmin, True)
    result = _clip_line(result, 0, xmax, False)
    result = _clip_line(result, 1, ymin, True)
    result = _clip_line(result, 1, ymax, False)
    return result


def _polygon_area(points: Sequence) -> Fraction:
    if len(points) < 3:
        return Fraction(0)
    total = Fraction(0)
    for (x1, y1), (x2, y2) in zip(points, tuple(points[1:]) + (points[0],)):
        total += x1 * y2 - x2 * y1
    return abs(total) / 2


def _determinant3(matrix: Sequence[Sequence[Fraction]]) -> Fraction:
    return (matrix[0][0] * (matrix[1][1] * matrix[2][2]
                            - matrix[1][2] * matrix[2][1])
            - matrix[0][1] * (matrix[1][0] * matrix[2][2]
                              - matrix[1][2] * matrix[2][0])
            + matrix[0][2] * (matrix[1][0] * matrix[2][1]
                              - matrix[1][1] * matrix[2][0]))


def _exact_plane_basis(axis: int, calibration: "RegionSupportCalibration",
                       ) -> tuple[tuple[Fraction, ...], ...]:
    """The retained calibration's plane basis as exact rational vectors.

    This is the rational counterpart of ``lattice.plane_basis`` and preserves its
    tangent convention exactly: for ``axis`` the normal is row ``axis`` of the
    calibration axis matrix and the two in-plane directions are the remaining rows
    in ascending label order ``others = [i for i in range(3) if i != axis]``
    (first ``others[0]``, second ``others[1]``).  Each declared ``float64`` entry
    is an exact dyadic rational converted once with :class:`Fraction` *before* any
    arithmetic, so no rounded intermediate can enter the basis.
    """
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    others = [index for index in range(3) if index != axis]
    rows = calibration.axis_matrix

    def _rational(row: int) -> tuple[Fraction, ...]:
        return tuple(Fraction(float(value)) for value in rows[row])

    return _rational(axis), _rational(others[0]), _rational(others[1])


def _exact_homography(axis: int, calibration: "RegionSupportCalibration",
                      alternative: "RetainedAlternative",
                      ) -> tuple[tuple[Fraction, ...], ...]:
    """Exact rational homography built from the declared calibration inputs.

    The accepted closed form is ``projection @ basis`` with
    ``basis = column_stack(first, second, distance*side*normal)`` and
    ``normal, first, second = plane_basis(axis, axis_matrix)``.

    The composition is exact from the retained calibration and alternative: the
    plane basis and the pinhole projection are each built as exact rationals from
    the declared ``float64`` inputs (:func:`_exact_plane_basis`), then multiplied
    and accumulated with :class:`Fraction` only, so no coefficient is ever formed
    as a rounded ``float64`` product and later converted.  The retained scale is
    likewise the exact rational product ``side * distance_blocks``.  This is the
    only homography the cell geometry consumes;
    :func:`lattice_planes.rectification_homography` is consulted separately for
    finiteness validation and its rounded coefficients never reach a cell
    boundary.
    """
    normal, first, second = _exact_plane_basis(int(axis), calibration)
    n, f, s = normal, first, second
    focal = Fraction(calibration.focal)
    centre_x = Fraction(calibration.principal_point[0])
    centre_y = Fraction(calibration.principal_point[1])
    scale = Fraction(calibration.side) * Fraction(alternative.distance_blocks)
    basis = (
        (f[0], s[0], scale * n[0]),
        (f[1], s[1], scale * n[1]),
        (f[2], s[2], scale * n[2]),
    )
    projection = (
        (focal, Fraction(0), centre_x),
        (Fraction(0), -focal, centre_y),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    return tuple(
        tuple(sum((projection[i][k] * basis[k][j] for k in range(3)),
                  Fraction(0)) for j in range(3))
        for i in range(3))


def _project(homography: Sequence[Sequence[Fraction]], u: Fraction,
             v: Fraction) -> tuple[Fraction, Fraction] | None:
    denominator = (homography[2][0] * u + homography[2][1] * v
                   + homography[2][2])
    if denominator <= 0:
        return None
    x = (homography[0][0] * u + homography[0][1] * v
         + homography[0][2]) / denominator
    y = (homography[1][0] * u + homography[1][1] * v
         + homography[1][2]) / denominator
    return (x, y)


def _excluded_area(polygon: Sequence, mask: np.ndarray, counter: _SupportCounter,
                   kind: str) -> Fraction:
    """Area of ``polygon`` covered by mask-False pixels (outside the image too)."""
    if not polygon:
        return Fraction(0)
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    height, width = int(mask.shape[0]), int(mask.shape[1])
    x_lo = max(0, math.floor(min(xs)))
    x_hi = min(width - 1, math.ceil(max(xs)) - 1)
    y_lo = max(0, math.floor(min(ys)))
    y_hi = min(height - 1, math.ceil(max(ys)) - 1)
    total = Fraction(0)
    for pixel_y in range(y_lo, y_hi + 1):
        for pixel_x in range(x_lo, x_hi + 1):
            if kind == "region":
                counter.region_pixel_tests += 1
            else:
                counter.usable_pixel_tests += 1
            counter.check()
            if not bool(mask[pixel_y, pixel_x]):
                clipped = _clip_rect(polygon, Fraction(pixel_x),
                                     Fraction(pixel_x + 1), Fraction(pixel_y),
                                     Fraction(pixel_y + 1))
                counter.polygon_vertices += len(clipped)
                total += _polygon_area(clipped)
    return total


def _rasterize(*, region_mask: np.ndarray, usable_mask: np.ndarray,
               image_shape: tuple[int, int],
               homography: tuple[tuple[Fraction, ...], ...],
               origin: tuple[Fraction, Fraction],
               grid_shape: tuple[int, int],
               counter: _SupportCounter) -> tuple[AlignedSupportCell, ...]:
    rows, columns = grid_shape
    height, width = image_shape
    matrix = homography
    origin_u, origin_v = origin
    cells: list[AlignedSupportCell] = []
    for row in range(rows):
        for column in range(columns):
            counter.cells_examined += 1
            counter.check()
            u_start = origin_u + column
            v_start = origin_v + row
            corners = (
                (u_start, v_start), (u_start + 1, v_start),
                (u_start + 1, v_start + 1), (u_start, v_start + 1),
            )
            counter.polygon_vertices += 4
            counter.check()
            projected: list[tuple[Fraction, Fraction]] = []
            degenerate = False
            for u, v in corners:
                point = _project(matrix, u, v)
                if point is None:
                    degenerate = True
                    break
                projected.append(point)
            if degenerate:
                cell_area = Fraction(0)
                region_excluded = Fraction(0)
                usable_excluded = Fraction(0)
                frame_excluded = Fraction(0)
            else:
                cell_area = _polygon_area(projected)
                frame_polygon = _clip_rect(projected, Fraction(0), Fraction(width),
                                           Fraction(0), Fraction(height))
                counter.frame_clips += 1
                counter.polygon_vertices += len(frame_polygon)
                counter.check()
                frame_excluded = cell_area - _polygon_area(frame_polygon)
                region_excluded = frame_excluded + _excluded_area(
                    frame_polygon, region_mask, counter, "region")
                usable_excluded = frame_excluded + _excluded_area(
                    frame_polygon, usable_mask, counter, "usable")
            cell = AlignedSupportCell.build(
                row=row, column=column, cell_area=cell_area,
                region_excluded=region_excluded,
                usable_excluded=usable_excluded,
                frame_excluded=frame_excluded)
            if cell.state == "supported":
                counter.supported_cells += 1
            else:
                counter.unknown_cells += 1
            counter.check()
            cells.append(cell)
    return tuple(cells)


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def _calibration_from_reading(
        reading: "lattice_planes.PlaneReading") -> RegionSupportCalibration:
    rectification = reading.rectification
    if not isinstance(rectification, lattice_planes.PlaneRectification):
        raise ValueError("reading has no rectification calibration")
    return RegionSupportCalibration.build(
        axis_matrix=rectification.axis_matrix, focal=rectification.focal,
        principal_point=rectification.principal_point,
        centre=rectification.centre, side=reading.side)


def _select_alternative(reading: "lattice_planes.PlaneReading",
                        blocks_per_period: Any) -> RetainedAlternative:
    try:
        matches = reading.matching_alternatives(blocks_per_period)
    except (TypeError, ValueError, OverflowError):
        matches = ()
    if len(matches) != 1:
        raise ValueError("alternative_not_retained")
    return RetainedAlternative.build(matches[0])


def _canonical_origin(origin_uv: Any) -> tuple[Fraction, Fraction]:
    if not isinstance(origin_uv, (tuple, list)) or len(origin_uv) != 2:
        raise ValueError("origin must be a pair")
    return (_fraction(origin_uv[0]), _fraction(origin_uv[1]))


def _validate_grid(grid_shape: Any) -> tuple[int, int]:
    if (not isinstance(grid_shape, (tuple, list)) or len(grid_shape) != 2
            or any(type(value) is not int or value <= 0 for value in grid_shape)):
        raise ValueError("grid shape must be a pair of positive exact integers")
    return (int(grid_shape[0]), int(grid_shape[1]))


def build_aligned_region_support(
    *,
    reading: "lattice_planes.PlaneReading",
    reading_identity: v2a.ReadingIdentity,
    reading_universe: Sequence["lattice_planes.PlaneReading"],
    usable_mask: np.ndarray,
    image_sha256: str,
    blocks_per_period: Any,
    origin_uv: Any,
    grid_shape: Any,
    budget: AlignedSupportBudget,
    phase_anchor_sha256: str | None = None,
    producer_version: str = V1_SUPPORT_PRODUCER_VERSION,
) -> AlignedRegionSupport | AlignedSupportRefusal:
    """Build one isolated, provenance-bound phase-aligned support map.

    ``reading_universe`` is the complete canonical reading universe the supplied
    V2-A ``reading_identity`` was derived from; it is required so the identity
    can be rebound to the reading's real content instead of being trusted as an
    opaque digest.  Validation is deterministic and ordered: the first failing
    precondition returns its typed refusal.  All input and projection validation
    (including a non-finite or degenerate homography) completes before any
    budget decision, so an invalid request never reports ``budget_exhausted``.
    Budget excess is atomic: every partially examined cell is discarded and a
    ``budget_exhausted`` refusal is returned.
    """
    if not isinstance(budget, AlignedSupportBudget):
        return _make_refusal("invalid_input", {"field": "budget"})
    counter = _SupportCounter(budget)
    try:
        if type(producer_version) is not str \
                or producer_version != V1_SUPPORT_PRODUCER_VERSION:
            raise _InputRefusal("invalid_input", {"field": "producer_version"})
        if not isinstance(reading, lattice_planes.PlaneReading):
            raise _InputRefusal("invalid_input", {"field": "reading"})
        if not _hex64(image_sha256):
            raise _InputRefusal("invalid_input", {"field": "image_sha256"})
        if phase_anchor_sha256 is not None and not _hex64(phase_anchor_sha256):
            raise _InputRefusal("invalid_input", {"field": "phase_anchor_sha256"})
        if not isinstance(reading_identity, v2a.ReadingIdentity):
            raise _InputRefusal("missing_provenance", {"field": "reading_identity"})
        evidence = reading.region_evidence
        if not isinstance(evidence, lattice_planes.PlaneRegionEvidence):
            raise _InputRefusal("missing_provenance", {"field": "region_evidence"})
        try:
            region_mask = lattice_planes._decode_region_evidence(
                evidence, reading.axis, reading.polygon)
        except (ValueError, TypeError) as error:
            raise _InputRefusal("invalid_region_mask",
                                {"detail": str(error)[:160]}) from error
        _validate_identity_content(reading, reading_identity, reading_universe)
        if (not isinstance(usable_mask, np.ndarray) or usable_mask.ndim != 2
                or usable_mask.dtype != np.dtype(bool) or usable_mask.size == 0
                or not bool(usable_mask.any())):
            raise _InputRefusal("invalid_usable_mask", {"field": "usable_mask"})
        image_shape = (int(evidence.image_shape[0]), int(evidence.image_shape[1]))
        if tuple(int(value) for value in usable_mask.shape) != image_shape:
            raise _InputRefusal("image_shape_mismatch", {
                "evidence": list(image_shape),
                "usable": [int(usable_mask.shape[0]), int(usable_mask.shape[1])]})
        try:
            calibration = _calibration_from_reading(reading)
        except (ValueError, TypeError, AttributeError) as error:
            raise _InputRefusal("calibration_mismatch",
                                {"detail": str(error)[:160]}) from error
        try:
            alternative = _select_alternative(reading, blocks_per_period)
        except (ValueError, TypeError) as error:
            raise _InputRefusal("alternative_not_retained",
                                {"detail": str(error)[:160]}) from error
        try:
            origin = _canonical_origin(origin_uv)
        except (ValueError, TypeError) as error:
            raise _InputRefusal("invalid_origin",
                                {"detail": str(error)[:160]}) from error
        try:
            rows, columns = _validate_grid(grid_shape)
        except (ValueError, TypeError) as error:
            raise _InputRefusal("invalid_grid",
                                {"detail": str(error)[:160]}) from error
        # Projection validation precedes every budget decision: an overflowing
        # or degenerate homography is an input/calibration refusal even when the
        # requested grid would exceed a small ``max_cells`` cap.
        try:
            float_homography = lattice_planes.rectification_homography(
                reading.axis, calibration.axis_matrix, calibration.focal,
                calibration.principal_point, calibration.side,
                alternative.distance_blocks)
        except (ValueError, TypeError, OverflowError,
                np.linalg.LinAlgError) as error:
            raise _InputRefusal("calibration_mismatch",
                                {"detail": str(error)[:160]}) from error
        try:
            finite = bool(np.isfinite(
                np.asarray(float_homography, dtype=float)).all())
        except (TypeError, ValueError, OverflowError):
            finite = False
        if not finite:
            raise _InputRefusal("calibration_mismatch",
                                {"detail": "non-finite homography"})
        try:
            homography = _exact_homography(reading.axis, calibration,
                                           alternative)
        except (ValueError, TypeError, AttributeError, OverflowError) as error:
            raise _InputRefusal("calibration_mismatch",
                                {"detail": str(error)[:160]}) from error
        # The composition is exact by construction; refuse rather than let a
        # non-rational (rounded) coefficient ever reach a cell boundary.
        if any(type(value) is not Fraction for row in homography for value in row):
            raise _InputRefusal("calibration_mismatch",
                                {"detail": "homography is not exact rational"})
        if _determinant3(homography) == 0:
            raise _InputRefusal("calibration_mismatch",
                                {"detail": "degenerate homography"})
        tangent_labels = tuple(label for label in range(3)
                               if label != int(reading.axis))
        if len(tangent_labels) != 2:
            raise _InputRefusal("invalid_input", {"field": "reading.axis"})
        if rows * columns > budget.max_cells:
            raise _BudgetStop("max_cells", budget.max_cells, rows * columns)
        cells = _rasterize(
            region_mask=region_mask, usable_mask=usable_mask,
            image_shape=image_shape, homography=homography, origin=origin,
            grid_shape=(rows, columns), counter=counter)
        work = counter.snapshot()
        return AlignedRegionSupport.build(
            schema_version=V1_SUPPORT_SCHEMA,
            producer_version=V1_SUPPORT_PRODUCER_VERSION,
            coverage_policy=V1_SUPPORT_COVERAGE_POLICY,
            source_region_id=str(evidence.source_region_id),
            region_mask_sha256=str(evidence.mask_sha256),
            image_sha256=str(image_sha256),
            image_shape=image_shape,
            usable_mask_sha256=usable_mask_sha256(usable_mask),
            reading_identity=reading_identity,
            reading_identity_sha256=reading_identity.identity_sha256,
            calibration=calibration,
            alternative=alternative,
            origin_uv=(_pair(origin[0]), _pair(origin[1])),
            grid_shape=(rows, columns),
            row_label=int(tangent_labels[1]),
            column_label=int(tangent_labels[0]),
            row_convention=_ROW_CONVENTION,
            column_convention=_COLUMN_CONVENTION,
            phase_anchor_sha256=phase_anchor_sha256,
            cells=cells,
            work=work)
    except _BudgetStop as stop:
        return _make_refusal("budget_exhausted", {
            "cap": stop.cap, "limit": int(stop.limit),
            "observed": int(stop.observed)}, counter.snapshot())
    except _InputRefusal as refusal:
        return _make_refusal(refusal.reason, refusal.detail)
