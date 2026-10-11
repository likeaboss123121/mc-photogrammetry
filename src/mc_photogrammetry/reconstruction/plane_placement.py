"""M4 V2 isolated resolved face placement (gauge-relative, replay-gated).

This module implements only the isolated V2 step of
``docs/reports/M4-placement-volume-bridge-contract-2026-09-19.md``: resolve one
reading's *normal face* and its supported lateral cells onto exact integer block
coordinates in the accepted C2-R-B canonical gauge, using replay-validated
V2-A/V2-B/C2-R-B evidence and the shared :func:`plane_graph.axis_transform`.

It defines no volume, no ``BlockVolume``, no block states or material inference,
no M5 state fusion, no CLI/UI, no persistence schema, no litematic wiring, no
promotion gate and no manual-coordinate input.  Every numeric input is
image-derived pipeline evidence; no world position, direction, bound, manifest,
F3 value or other advanced-only metadata is consumed.

Load-bearing rules:

* **One genuine universe.**  Placement consumes exactly one
  :class:`PlacementUniverse`: one canonical reading universe whose C2-R-B result
  is a single connected, exhaustive, cycle-validated, ``unique_validated``
  component.  A frame may carry several normal-axis readings, so that single
  component may span *mixed* world normal axes; V2-B must determine all three
  world axes from the one component.  Separate partial components are never
  combined, sliced, or aggregated into a fake certificate.
* **Exact signed face formula** of the V2 prerequisite audit
  (``DEEPSEEK-M4-V2-placement-prerequisite-audit-2026-09-13`` lines 24-42) and
  the bridge contract, applied literally with ``c = side``,
  ``d = distance_blocks`` and ``(n, sn) = axis_transform(a, q)``::

      face_gauge   = T[n] + sn * c * d
      F            = face_gauge - alpha[n]
      outward_sign = -sn * c
      normal_block = F - 1 if outward_sign > 0 else F

  ``T`` is the full signed reduced-dyadic C2-R-B certificate translation for the
  reading's frame (integers and negative fractions preserved, never reduced
  modulo one and never substituted by a supplied world position); ``alpha[n]`` is
  the V2-B determined seam offset for world axis ``n``.  ``F`` must be an exact
  integer, otherwise the atomic ``face_integer_unresolved`` refusal is returned.
* **Exact signed tangent formula.**  The two tangent labels are the sorted axes
  excluding the reading's normal label, each mapped by the same transform.  For a
  tangent local label ``l`` with ``(m, sm) = axis_transform(l, q)`` the world
  coordinate of the local cell-grid origin is ``T[m] + sm*origin_l - alpha[m]``,
  which must be an exact integer.  A supported unit cell at local index ``i``
  reports the *lower* world interval endpoint: ``anchor + sm*i`` for ``sm > 0``
  and ``anchor - (i + 1)`` for ``sm < 0`` (the negative-sign adjustment).  Every
  tangent mapping and every placed cell is re-derived on construction, so a
  freshly resealed lateral anchor cannot contradict its parent face.
* **Bound supports.**  Every :class:`~plane_region_support.AlignedRegionSupport`
  is bound to the single V2-A endpoint assignment, the reading's selected
  retained alternative, the reading's calibration, its original region-mask and
  image digests, and the accepted anchor digest; it is *rebuilt* from the
  original reading, canonical universe, usable mask and support budget and must
  reproduce byte-for-byte before any of its cells are consumed.  A distance
  1 -> 2 reseal, a foreign phase-anchor digest or a resealed support is refused.
* **Atomic truthful budgets.**  Every counter is incremented exactly when its
  unit of work is performed and the refusal reports that observed value; a
  ``budget_exhausted`` refusal never crashes and never carries partial faces.
  Invalid inputs are rejected before any budget decision.
* **Every required ``(frame, world axis)`` translation** must be *supported* by
  the accepted C1 axis support (``connected_to_global_anchor`` and no refusal),
  so a least-squares min-norm ``0.0`` fill is never accepted as evidence.
* The whole-build integer translation gauge stays explicitly unresolved and is
  carried as typed state, never silently fixed to an absolute world origin.

The module performs no IO and has no filesystem, network, process, environment,
or dynamic-import access.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence

from mc_photogrammetry.reconstruction import lattice_planes
from mc_photogrammetry.reconstruction import plane_endpoint_provenance as v2a
from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction import plane_phase_anchor as v2b
from mc_photogrammetry.reconstruction import plane_region_support as v1
from mc_photogrammetry.reconstruction import plane_relative_translation


__all__ = [
    "V2_PLACEMENT_SCHEMA",
    "V2_PLACEMENT_PRODUCER_VERSION",
    "V2_PLACEMENT_GAUGE_STATE",
    "V2_PLACEMENT_WORLD_AXES",
    "V2_PLACEMENT_REFUSAL_REASONS",
    "PlacementBudget",
    "PlacementWork",
    "PlacementRefusal",
    "PlacementUniverse",
    "TangentAxisMapping",
    "PlacedCell",
    "ResolvedFace",
    "ResolvedFacePlacement",
    "make_placement_budget",
    "placement_anchor_digest",
    "face_axis_coordinate",
    "tangent_axis_coordinate",
    "placement_sort_key",
    "canonical_placement_order",
    "resolve_face_placement",
]

V2_PLACEMENT_SCHEMA = "m4-v2-resolved-face-placement-2"
V2_PLACEMENT_PRODUCER_VERSION = "v2-face-placement-producer-2"
V2_PLACEMENT_GAUGE_STATE = "unresolved_whole_build_integer_translation"
V2_PLACEMENT_WORLD_AXES: tuple[int, ...] = (0, 1, 2)
V2_PLACEMENT_LOCAL_AXES: tuple[str, ...] = ("u", "v")
V2_PLACEMENT_REFUSAL_REASONS: tuple[str, ...] = (
    "invalid_input",
    "missing_provenance",
    "provenance_mismatch",
    "replay_mismatch",
    "c2rb_gate_not_unique",
    "endpoint_gate_not_unique",
    "anchor_gate_not_unique",
    "anchor_axis_unresolved",
    "anchor_conflict",
    "translation_unsupported",
    "alternative_not_retained",
    "support_mismatch",
    "support_replay_mismatch",
    "axis_transform_invalid",
    "cell_contradiction",
    "face_integer_unresolved",
    "budget_exhausted",
    "quadrant_constraint_inconsistent",
)


# ---------------------------------------------------------------------------
# Canonical JSON and exact-rational helpers
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


def placement_anchor_digest(alpha: Sequence[tuple[int, int]]) -> str:
    """Canonical digest binding the three accepted reduced-dyadic anchors.

    V1 supports must carry exactly this digest; a support resealed with any
    other (foreign) phase-anchor digest is refused, so a support cannot be
    swapped between two equally replay-valid anchor vectors.
    """
    if (not isinstance(alpha, Sequence) or len(alpha) != 3):
        raise ValueError("anchor vector must have three entries")
    payload = []
    for item in alpha:
        value = _canonical_dyadic(tuple(item))
        payload.append([value.numerator, value.denominator])
    return _sha256(payload)


def _signed_dyadic(value: Any) -> Fraction:
    """Parse one C2-R-B certificate translation entry (full signed dyadic).

    Exactly the domain of ``plane_relative_translation._validate_dyadic``: a
    pair of exact ``int``s with a positive power-of-two denominator and
    ``gcd(abs(numerator), denominator) == 1``.  Neither a sign reduction nor a
    modulo-one reduction is performed here; periodicity is only ever applied at
    the exact anchor, never at parse time.
    """
    if (not isinstance(value, tuple) or len(value) != 2
            or any(type(part) is not int for part in value)):
        raise ValueError("noncanonical_dyadic")
    numerator, denominator = value
    if (denominator <= 0 or denominator & (denominator - 1)
            or math.gcd(abs(numerator), denominator) != 1):
        raise ValueError("noncanonical_dyadic")
    return Fraction(numerator, denominator)


def _canonical_dyadic(value: Any) -> Fraction:
    """Parse one V2-B reduced dyadic anchor in ``[0, 1)``."""
    if (not isinstance(value, tuple) or len(value) != 2
            or any(type(part) is not int for part in value)):
        raise ValueError("noncanonical_dyadic")
    numerator, denominator = value
    if numerator == 0:
        if denominator != 1:
            raise ValueError("noncanonical_dyadic")
        return Fraction(0)
    if (denominator <= 0 or denominator & (denominator - 1)
            or numerator < 0 or numerator >= denominator or numerator % 2 == 0):
        raise ValueError("noncanonical_dyadic")
    return Fraction(numerator, denominator)


def _reduce(value: Fraction) -> tuple[int, int]:
    value = Fraction(value)
    if value == 0:
        return (0, 1)
    return (value.numerator, value.denominator)


def _rational(value: Any) -> Fraction:
    """Parse a caller-facing exact lattice rational (int, float or pair)."""
    if isinstance(value, Fraction):
        return Fraction(value)
    if type(value) is int and type(value) is not bool:
        return Fraction(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("rational must be finite")
        return Fraction(*value.as_integer_ratio())
    if (isinstance(value, (tuple, list)) and len(value) == 2
            and all(type(part) is int for part in value)):
        if value[1] <= 0:
            raise ValueError("rational denominator must be positive")
        return Fraction(value[0], value[1])
    raise ValueError("rational must be an int, float or (numerator, denominator)")


def _finite(value: Any) -> float | None:
    if type(value) not in (int, float) or type(value) is bool:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


# ---------------------------------------------------------------------------
# Budget, work and refusal
# ---------------------------------------------------------------------------

_BUDGET_FIELDS: tuple[str, ...] = (
    "max_faces",
    "max_supports_checked",
    "max_support_cells",
    "max_frame_axis_lookups",
    "max_integer_checks",
    "max_anchors",
)
_WORK_FIELDS: tuple[str, ...] = (
    "faces",
    "supports_checked",
    "support_cells",
    "frame_axis_lookups",
    "integer_checks",
    "anchors",
)
_WORK_TO_CAP = dict(zip(_WORK_FIELDS, _BUDGET_FIELDS))
_CAP_TO_WORK = {cap: work for work, cap in _WORK_TO_CAP.items()}


@dataclass(frozen=True, slots=True)
class PlacementBudget:
    """Explicit positive whole-request limits.  No limit has a default."""

    max_faces: int
    max_supports_checked: int
    max_support_cells: int
    max_frame_axis_lookups: int
    max_integer_checks: int
    max_anchors: int
    budget_sha256: str

    def __post_init__(self) -> None:
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 1
               for name in _BUDGET_FIELDS):
            raise ValueError("placement budget limits must be positive integers")
        if (not _hex64(self.budget_sha256)
                or self.budget_sha256 != _sha256(self._payload())):
            raise ValueError("placement budget digest mismatch")

    def _payload(self) -> dict:
        return {name: getattr(self, name) for name in _BUDGET_FIELDS}

    def to_record(self) -> dict:
        record = self._payload()
        record["budget_sha256"] = self.budget_sha256
        return record


def make_placement_budget(**caps: int) -> PlacementBudget:
    """Build a self-validating placement budget from explicit caps."""
    payload: dict[str, int] = {}
    for name in _BUDGET_FIELDS:
        if name not in caps:
            raise ValueError("placement budget is missing a limit")
        value = caps[name]
        if type(value) is not int or value < 1:
            raise ValueError("placement budget limits must be positive integers")
        payload[name] = value
    return PlacementBudget(**payload, budget_sha256=_sha256(payload))


@dataclass(frozen=True, slots=True)
class PlacementWork:
    """Exact observed placement work counters."""

    faces: int
    supports_checked: int
    support_cells: int
    frame_axis_lookups: int
    integer_checks: int
    anchors: int
    work_sha256: str

    def __post_init__(self) -> None:
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 0
               for name in _WORK_FIELDS):
            raise ValueError("placement work counters must be non-negative integers")
        if (not _hex64(self.work_sha256)
                or self.work_sha256 != _sha256(self._payload())):
            raise ValueError("placement work digest mismatch")

    def _payload(self) -> dict:
        return {name: getattr(self, name) for name in _WORK_FIELDS}

    def to_record(self) -> dict:
        record = self._payload()
        record["work_sha256"] = self.work_sha256
        return record


def _make_work(counters: Mapping[str, int]) -> PlacementWork:
    payload: dict[str, int] = {}
    for name in _WORK_FIELDS:
        value = counters.get(name, 0)
        if type(value) is not int or value < 0:
            raise ValueError("placement work counters must be non-negative integers")
        payload[name] = value
    return PlacementWork(**payload, work_sha256=_sha256(payload))


@dataclass(frozen=True, slots=True)
class PlacementRefusal:
    """One typed, atomic placement refusal.  It carries no faces or cells."""

    reason: str
    detail: dict
    detail_sha256: str
    work: PlacementWork | None
    refusal_sha256: str

    def __post_init__(self) -> None:
        if self.reason not in V2_PLACEMENT_REFUSAL_REASONS:
            raise ValueError("placement refusal reason is not in the closed enum")
        normalized = _json_value(self.detail)
        if not isinstance(normalized, dict):
            raise ValueError("placement refusal detail must be a JSON object")
        encoded = _canonical_json(normalized)
        if self.detail_sha256 != hashlib.sha256(encoded.encode("utf-8")).hexdigest():
            raise ValueError("placement refusal detail digest mismatch")
        if self.work is not None and not isinstance(self.work, PlacementWork):
            raise ValueError("placement refusal work is malformed")
        if "faces" in normalized or "cells" in normalized:
            raise ValueError("placement refusal must not embed partial faces")
        if self.reason == "budget_exhausted":
            if self.work is None or set(normalized) != {"cap", "limit", "observed"}:
                raise ValueError("placement budget refusal is malformed")
            cap = normalized["cap"]
            limit = normalized["limit"]
            observed = normalized["observed"]
            if (cap not in _CAP_TO_WORK or type(limit) is not int
                    or type(observed) is not int or limit < 1
                    or observed <= limit):
                raise ValueError("placement budget refusal evidence is malformed")
            if getattr(self.work, _CAP_TO_WORK[cap]) != observed:
                raise ValueError(
                    "placement budget refusal disagrees with its work counter")
        elif self.work is not None:
            raise ValueError("non-budget placement refusal must not carry work")
        payload = {
            "reason": self.reason,
            "detail_sha256": self.detail_sha256,
            "work": None if self.work is None else self.work.to_record(),
        }
        if (not _hex64(self.refusal_sha256)
                or self.refusal_sha256 != _sha256(payload)):
            raise ValueError("placement refusal digest mismatch")
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
                  work: PlacementWork | None = None) -> PlacementRefusal:
    if reason not in V2_PLACEMENT_REFUSAL_REASONS:
        reason = "invalid_input"
    normalized = _json_value({} if detail is None else detail)
    detail_sha = hashlib.sha256(
        _canonical_json(normalized).encode("utf-8")).hexdigest()
    payload = {
        "reason": reason,
        "detail_sha256": detail_sha,
        "work": None if work is None else work.to_record(),
    }
    return PlacementRefusal(reason, normalized, detail_sha, work,
                            _sha256(payload))


class _InputRefusal(Exception):
    def __init__(self, reason: str, detail: Any):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


class _BudgetStop(Exception):
    def __init__(self, cap: str, limit: int, observed: int):
        super().__init__(cap)
        self.cap = cap
        self.limit = int(limit)
        self.observed = int(observed)


class _Counter:
    """Truthful, checked placement work counters.

    ``tick`` increments exactly one counter and then checks *all* caps against
    the observed counters; a stop therefore always reports the exact observed
    value of the cap that was exceeded, never a stale or fabricated number.
    """

    __slots__ = ("budget", "faces", "supports_checked", "support_cells",
                 "frame_axis_lookups", "integer_checks", "anchors")

    def __init__(self, budget: PlacementBudget):
        self.budget = budget
        self.faces = 0
        self.supports_checked = 0
        self.support_cells = 0
        self.frame_axis_lookups = 0
        self.integer_checks = 0
        self.anchors = 0

    def tick(self, field: str, amount: int = 1) -> None:
        if field not in _WORK_FIELDS or type(amount) is not int or amount < 1:
            raise ValueError("invalid placement counter update")
        setattr(self, field, getattr(self, field) + amount)
        self.check()

    def check(self) -> None:
        for work_name, cap_name in _WORK_TO_CAP.items():
            observed = getattr(self, work_name)
            limit = getattr(self.budget, cap_name)
            if observed > limit:
                raise _BudgetStop(cap_name, limit, observed)

    def snapshot(self) -> PlacementWork:
        return _make_work({name: getattr(self, name) for name in _WORK_FIELDS})


# ---------------------------------------------------------------------------
# Pure signed face and tangent arithmetic (the exact contract formulas)
# ---------------------------------------------------------------------------

def face_axis_coordinate(
    *,
    local_label: int,
    quadrant: int,
    c: Any,
    d: Any,
    translation_dyadic: tuple[int, int],
    alpha_dyadic: tuple[int, int],
) -> dict:
    """Apply the exact signed face formula to one local label.

    ``(n, sn) = axis_transform(local_label, quadrant)``;
    ``face_gauge = T[n] + sn*c*d``; ``F = face_gauge - alpha[n]``;
    ``outward_sign = -sn*c``; ``normal_block = F - 1 if outward_sign > 0 else F``.

    Returns a record with the exact ``face_gauge`` and, only when ``F`` is an
    exact integer, the integer ``face`` and ``normal_block``.  A non-integral
    ``F`` is reported with ``face=None`` and ``integer=False`` so the caller can
    return the typed atomic ``face_integer_unresolved`` refusal without ever
    rounding.
    """
    world_axis, world_sign = plane_graph.axis_transform(
        int(local_label), int(quadrant))
    translation = _signed_dyadic(tuple(translation_dyadic))
    alpha = _canonical_dyadic(tuple(alpha_dyadic))
    c_value = _rational(c)
    d_value = _rational(d)
    face_gauge = translation + world_sign * c_value * d_value
    residual = face_gauge - alpha
    outward = -world_sign * c_value
    integral = residual.denominator == 1
    face = int(residual) if integral else None
    normal_block = None
    if integral:
        normal_block = face - 1 if outward > 0 else face
    return {
        "local_label": int(local_label),
        "quadrant": int(quadrant),
        "world_axis": world_axis,
        "world_sign": world_sign,
        "c": _reduce(c_value),
        "d": _reduce(d_value),
        "translation": list(_reduce(translation)),
        "alpha": list(_reduce(alpha)),
        "face_gauge": list(_reduce(face_gauge)),
        "residual": list(_reduce(residual)),
        "integer": integral,
        "face": face,
        "outward_sign": int(math.copysign(1, float(outward))) if outward != 0 else 0,
        "outward_raw": _reduce(outward),
        "normal_block": normal_block,
    }


def tangent_axis_coordinate(
    *,
    local_label: int,
    quadrant: int,
    origin: Any,
    translation_dyadic: tuple[int, int],
    alpha_dyadic: tuple[int, int],
) -> dict:
    """Apply the exact signed tangent formula to one tangent local label.

    ``(m, sm) = axis_transform(local_label, quadrant)``;
    ``anchor = T[m] + sm*origin - alpha[m]`` must be an exact integer.  The
    lower world interval endpoint of a unit cell at local index ``i`` is then
    ``anchor + sm*i`` for ``sm > 0`` and ``anchor - (i + 1)`` for ``sm < 0``
    (the negative-sign adjustment).  A non-integral anchor is reported with
    ``integer=False`` and ``anchor=None``; nothing is ever rounded.
    """
    world_axis, world_sign = plane_graph.axis_transform(
        int(local_label), int(quadrant))
    translation = _signed_dyadic(tuple(translation_dyadic))
    alpha = _canonical_dyadic(tuple(alpha_dyadic))
    origin_value = _rational(origin)
    raw = translation + world_sign * origin_value - alpha
    integral = raw.denominator == 1
    anchor = int(raw) if integral else None
    return {
        "local_label": int(local_label),
        "quadrant": int(quadrant),
        "world_axis": world_axis,
        "world_sign": world_sign,
        "origin": list(_reduce(origin_value)),
        "translation": list(_reduce(translation)),
        "alpha": list(_reduce(alpha)),
        "raw": list(_reduce(raw)),
        "integer": integral,
        "anchor": anchor,
    }


def _cell_lower_endpoint(anchor: int, world_sign: int, index: int) -> int:
    """Lower world interval endpoint of local unit cell ``index``."""
    if world_sign > 0:
        return anchor + index
    return anchor - (index + 1)


# ---------------------------------------------------------------------------
# Bound public records
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PlacementUniverse:
    """One complete accepted mixed-normal reading universe.

    It is the exact replay tuple of a single C2-R-B/V2-A/V2-B component: one
    canonical reading universe (a frame may carry several normal-axis readings)
    whose C2-R-B result is a single connected, exhaustive, cycle-validated
    translation class.  It carries the immutable upstream records, never a copy
    of their claimed values, plus the original usable masks needed to rebuild the
    V1 supports before consumption.
    """

    c2rb_result: Any
    vector_result: Any
    axis_support: Any
    p1d_result: Any
    matches: tuple
    phase_evidence: tuple
    readings: tuple
    signatures: tuple
    v2a_result: Any
    v2b_result: Any
    split_sources: tuple
    anchor_budget: Any
    support_budget: Any
    usable_masks: tuple
    comparison_tolerance: float
    closure_tolerance: float


def _face_payload(value: "ResolvedFace") -> dict:
    return {
        "frame": value.frame,
        "plane_index": value.plane_index,
        "reading_axis": value.reading_axis,
        "absolute_quadrant": value.absolute_quadrant,
        "source_region_id": value.source_region_id,
        "reading_identity_sha256": value.reading_identity_sha256,
        "support_sha256": value.support_sha256,
        "world_normal_axis": value.world_normal_axis,
        "world_normal_sign": value.world_normal_sign,
        "c": list(value.c),
        "d": list(value.d),
        "translation": [list(item) for item in value.translation],
        "alpha": [list(item) for item in value.alpha],
        "face_gauge": list(value.face_gauge),
        "face": value.face,
        "outward_sign": value.outward_sign,
        "normal_block": value.normal_block,
        "tangent_mappings": [mapping.to_record()
                             for mapping in value.tangent_mappings],
        "cells": [cell.to_record() for cell in value.cells],
    }


@dataclass(frozen=True, slots=True)
class TangentAxisMapping:
    """One tangent local label's exact world-axis mapping and integer anchor."""

    local_axis: str
    local_label: int
    world_axis: int
    world_sign: int
    origin: tuple[int, int]
    anchor: int
    mapping_sha256: str

    def __post_init__(self) -> None:
        if self.local_axis not in V2_PLACEMENT_LOCAL_AXES:
            raise ValueError("tangent mapping local axis is invalid")
        if type(self.local_label) is not int or self.local_label not in (0, 1, 2):
            raise ValueError("tangent mapping local label is invalid")
        if type(self.world_axis) is not int or self.world_axis not in (0, 1, 2):
            raise ValueError("tangent mapping world axis is invalid")
        if self.world_sign not in (-1, 1):
            raise ValueError("tangent mapping world sign is invalid")
        if not v1._validate_signed_pair(self.origin):
            raise ValueError("tangent mapping origin must be an exact rational pair")
        if type(self.anchor) is not int:
            raise ValueError("tangent mapping anchor must be an exact integer")
        if (not _hex64(self.mapping_sha256)
                or self.mapping_sha256 != _sha256(self._payload())):
            raise ValueError("tangent mapping digest mismatch")

    def _payload(self) -> dict:
        return {
            "local_axis": self.local_axis,
            "local_label": self.local_label,
            "world_axis": self.world_axis,
            "world_sign": self.world_sign,
            "origin": list(self.origin),
            "anchor": self.anchor,
        }

    def to_record(self) -> dict:
        record = self._payload()
        record["mapping_sha256"] = self.mapping_sha256
        return record


def _make_tangent_mapping(**fields: Any) -> TangentAxisMapping:
    payload = {
        "local_axis": fields["local_axis"],
        "local_label": fields["local_label"],
        "world_axis": fields["world_axis"],
        "world_sign": fields["world_sign"],
        "origin": fields["origin"],
        "anchor": fields["anchor"],
    }
    return TangentAxisMapping(**payload, mapping_sha256=_sha256(payload))


@dataclass(frozen=True, slots=True)
class PlacedCell:
    """One supported V1 cell with its exact integer lower world cell corners."""

    row: int
    column: int
    lateral_u: int
    lateral_v: int
    cell_sha256: str

    def __post_init__(self) -> None:
        if (type(self.row) is not int or self.row < 0
                or type(self.column) is not int or self.column < 0):
            raise ValueError("placed cell indices are invalid")
        if type(self.lateral_u) is not int or type(self.lateral_v) is not int:
            raise ValueError("placed cell world coordinates must be exact integers")
        if (not _hex64(self.cell_sha256)
                or self.cell_sha256 != _sha256(self._payload())):
            raise ValueError("placed cell digest mismatch")

    def _payload(self) -> dict:
        return {
            "row": self.row,
            "column": self.column,
            "lateral_u": self.lateral_u,
            "lateral_v": self.lateral_v,
        }

    def to_record(self) -> dict:
        record = self._payload()
        record["cell_sha256"] = self.cell_sha256
        return record


def _make_cell(**fields: Any) -> PlacedCell:
    payload = {
        "row": fields["row"],
        "column": fields["column"],
        "lateral_u": fields["lateral_u"],
        "lateral_v": fields["lateral_v"],
    }
    return PlacedCell(**payload, cell_sha256=_sha256(payload))


@dataclass(frozen=True, slots=True)
class ResolvedFace:
    """One reading's resolved gauge-relative face and its supported cells."""

    frame: str
    plane_index: int
    reading_axis: int
    absolute_quadrant: int
    source_region_id: str
    reading_identity_sha256: str
    support_sha256: str
    world_normal_axis: int
    world_normal_sign: int
    c: tuple[int, int]
    d: tuple[int, int]
    translation: tuple[tuple[int, int], ...]
    alpha: tuple[tuple[int, int], ...]
    face_gauge: tuple[int, int]
    face: int
    outward_sign: int
    normal_block: int
    tangent_mappings: tuple[TangentAxisMapping, ...]
    cells: tuple[PlacedCell, ...]
    face_sha256: str

    def __post_init__(self) -> None:
        if type(self.frame) is not str or not self.frame:
            raise ValueError("resolved face frame is invalid")
        if type(self.plane_index) is not int or self.plane_index < 0:
            raise ValueError("resolved face plane index is invalid")
        if type(self.reading_axis) is not int or self.reading_axis not in (0, 1, 2):
            raise ValueError("resolved face reading axis is invalid")
        if type(self.absolute_quadrant) is not int \
                or self.absolute_quadrant not in (0, 1, 2, 3):
            raise ValueError("resolved face quadrant is invalid")
        if not _hex64(self.source_region_id) \
                or not _hex64(self.reading_identity_sha256) \
                or not _hex64(self.support_sha256):
            raise ValueError("resolved face provenance digests are malformed")
        if type(self.world_normal_axis) is not int \
                or self.world_normal_axis not in (0, 1, 2):
            raise ValueError("resolved face world normal axis is invalid")
        if self.world_normal_sign not in (-1, 1):
            raise ValueError("resolved face world normal sign is invalid")
        # The normal face is the reading's own local axis mapped through the
        # shared discrete transform.  Binding it here means a resealed world
        # normal (e.g. axis 1 for local axis 0 at quadrant 0) cannot contradict
        # the quadrant it claims to have been resolved in.
        expected_normal = plane_graph.axis_transform(
            self.reading_axis, self.absolute_quadrant)
        if (self.world_normal_axis, self.world_normal_sign) != expected_normal:
            raise ValueError("resolved face normal does not replay its transform")
        if (not v1._validate_signed_pair(self.c)
                or not v1._validate_signed_pair(self.d)
                or not v1._validate_signed_pair(self.face_gauge)):
            raise ValueError("resolved face rationals are malformed")
        if self.c[1] <= 0 or self.d[0] <= 0:
            raise ValueError("resolved face side and distance must be positive")
        if (not isinstance(self.translation, tuple) or len(self.translation) != 3
                or any(not v1._validate_signed_pair(item)
                       for item in self.translation)):
            raise ValueError("resolved face translations are malformed")
        if (not isinstance(self.alpha, tuple) or len(self.alpha) != 3
                or any(not v1._validate_signed_pair(item) for item in self.alpha)):
            raise ValueError("resolved face anchors are malformed")
        if type(self.face) is not int or type(self.normal_block) is not int:
            raise ValueError("resolved face integer coordinates are invalid")
        if self.outward_sign not in (-1, 1):
            raise ValueError("resolved face outward sign is invalid")
        if (not isinstance(self.tangent_mappings, tuple)
                or len(self.tangent_mappings) != 2
                or any(not isinstance(item, TangentAxisMapping)
                       for item in self.tangent_mappings)
                or tuple(item.local_axis for item in self.tangent_mappings)
                != V2_PLACEMENT_LOCAL_AXES):
            raise ValueError("resolved face tangent mappings are malformed")
        if not isinstance(self.cells, tuple) \
                or any(not isinstance(cell, PlacedCell) for cell in self.cells):
            raise ValueError("resolved face cells are malformed")
        expected_order = tuple(sorted(
            ((cell.row, cell.column, cell.cell_sha256) for cell in self.cells)))
        if tuple((cell.row, cell.column, cell.cell_sha256)
                 for cell in self.cells) != expected_order \
                or len({(cell.row, cell.column) for cell in self.cells}) \
                != len(self.cells):
            raise ValueError("resolved face cells are not in canonical order")

        # Cross-check the retained normal formula: the face integer must satisfy
        # the exact gauge relation against the retained translation, side,
        # distance and anchor, and the normal block must be the outward-side
        # neighbour.
        translation = _signed_dyadic(self.translation[self.world_normal_axis])
        alpha = _canonical_dyadic(self.alpha[self.world_normal_axis])
        side = Fraction(*self.c)
        distance = Fraction(*self.d)
        gauge = (translation + self.world_normal_sign * side * distance)
        if gauge != Fraction(*self.face_gauge):
            raise ValueError("resolved face gauge does not replay its inputs")
        if gauge - alpha != self.face:
            raise ValueError("resolved face integer does not replay its anchor")
        outward = -self.world_normal_sign * side
        expected_sign = 1 if outward > 0 else -1
        expected_block = self.face - 1 if outward > 0 else self.face
        if self.outward_sign != expected_sign or self.normal_block != expected_block:
            raise ValueError("resolved face outward block does not replay")

        # Cross-check every tangent anchor and cell against the retained
        # translation, origin and anchor.  A freshly resealed lateral anchor
        # cannot survive this parent-child re-derivation.
        tangent_labels = tuple(label for label in V2_PLACEMENT_WORLD_AXES
                               if label != self.reading_axis)
        for mapping, expected_local in zip(self.tangent_mappings, tangent_labels):
            if mapping.local_label != expected_local:
                raise ValueError("resolved face tangent label is not canonical")
            expected_axis, expected_sign = plane_graph.axis_transform(
                mapping.local_label, self.absolute_quadrant)
            if (mapping.world_axis, mapping.world_sign) != (expected_axis, expected_sign):
                raise ValueError("resolved face tangent transform is inconsistent")
            axis_translation = _signed_dyadic(self.translation[expected_axis])
            axis_alpha = _canonical_dyadic(self.alpha[expected_axis])
            raw = (axis_translation
                   + expected_sign * Fraction(*mapping.origin) - axis_alpha)
            if raw.denominator != 1 or int(raw) != mapping.anchor:
                raise ValueError("resolved face tangent anchor does not replay")
        mapping_u, mapping_v = self.tangent_mappings
        for cell in self.cells:
            expected_u = _cell_lower_endpoint(
                mapping_u.anchor, mapping_u.world_sign, int(cell.column))
            expected_v = _cell_lower_endpoint(
                mapping_v.anchor, mapping_v.world_sign, int(cell.row))
            if (cell.lateral_u, cell.lateral_v) != (expected_u, expected_v):
                raise ValueError("resolved face cell contradicts its tangent anchor")

        if (not _hex64(self.face_sha256)
                or self.face_sha256 != _sha256(_face_payload(self))):
            raise ValueError("resolved face digest mismatch")

    @classmethod
    def build(cls, **fields: Any) -> "ResolvedFace":
        provisional = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "face_sha256", "")
        object.__setattr__(provisional, "face_sha256",
                           _sha256(_face_payload(provisional)))
        cls.__post_init__(provisional)
        return provisional

    def to_record(self) -> dict:
        record = _face_payload(self)
        record["face_sha256"] = self.face_sha256
        return record


def _placement_payload(value: "ResolvedFacePlacement") -> dict:
    return {
        "schema_version": value.schema_version,
        "producer_version": value.producer_version,
        "gauge_state": value.gauge_state,
        "gauge_axes": list(value.gauge_axes),
        "alpha": [list(item) for item in value.alpha],
        "component_bindings": [dict(binding) for binding in value.component_bindings],
        "faces": [face.to_record() for face in value.faces],
        "work": value.work.to_record(),
    }


@dataclass(frozen=True, slots=True)
class ResolvedFacePlacement:
    """The frozen, versioned, gauge-relative resolved face placement."""

    schema_version: str
    producer_version: str
    gauge_state: str
    gauge_axes: tuple[int, ...]
    alpha: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    component_bindings: tuple[dict, ...]
    faces: tuple[ResolvedFace, ...]
    work: PlacementWork
    placement_sha256: str

    def __post_init__(self) -> None:
        if (self.schema_version != V2_PLACEMENT_SCHEMA
                or self.producer_version != V2_PLACEMENT_PRODUCER_VERSION
                or self.gauge_state != V2_PLACEMENT_GAUGE_STATE):
            raise ValueError("placement record version or gauge is unsupported")
        if (not isinstance(self.gauge_axes, tuple)
                or self.gauge_axes != V2_PLACEMENT_WORLD_AXES):
            raise ValueError("placement gauge axes are invalid")
        if (not isinstance(self.alpha, tuple) or len(self.alpha) != 3
                or any(not v1._validate_signed_pair(item) for item in self.alpha)):
            raise ValueError("placement alpha vector is malformed")
        if (not isinstance(self.component_bindings, tuple)
                or not self.component_bindings
                or any(not isinstance(item, dict) for item in self.component_bindings)):
            raise ValueError("placement component bindings are malformed")
        if (not isinstance(self.faces, tuple)
                or any(not isinstance(face, ResolvedFace) for face in self.faces)):
            raise ValueError("placement faces are malformed")
        if tuple(face_sort_key(face) for face in self.faces) \
                != tuple(sorted(face_sort_key(face) for face in self.faces)):
            raise ValueError("placement faces are not in canonical order")
        for face in self.faces:
            for axis in V2_PLACEMENT_WORLD_AXES:
                if _canonical_dyadic(face.alpha[axis]) \
                        != _canonical_dyadic(self.alpha[axis]):
                    raise ValueError("placement face anchor disagrees with its vector")
        if not isinstance(self.work, PlacementWork):
            raise ValueError("placement work record is malformed")
        derived = _derived_work(self.faces)
        for name in ("faces", "support_cells", "anchors"):
            if getattr(self.work, name) != derived[name]:
                raise ValueError("placement work does not replay its faces")
        if self.work.supports_checked != len(self.faces):
            raise ValueError("placement support counter does not replay its faces")
        if (not _hex64(self.placement_sha256)
                or self.placement_sha256 != _sha256(_placement_payload(self))):
            raise ValueError("placement digest mismatch")

    @classmethod
    def build(cls, **fields: Any) -> "ResolvedFacePlacement":
        provisional = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "placement_sha256", "")
        object.__setattr__(provisional, "placement_sha256",
                           _sha256(_placement_payload(provisional)))
        cls.__post_init__(provisional)
        return provisional

    def to_record(self) -> dict:
        record = _placement_payload(self)
        record["placement_sha256"] = self.placement_sha256
        return record


def _derived_work(faces: Sequence[ResolvedFace]) -> dict:
    return {
        "faces": len(faces),
        "support_cells": sum(len(face.cells) for face in faces),
        "anchors": sum(len(face.tangent_mappings) + 2 * len(face.cells)
                       for face in faces),
    }


def face_sort_key(face: ResolvedFace) -> tuple:
    return (face.frame, face.plane_index, face.reading_axis,
            face.reading_identity_sha256, face.support_sha256)


def placement_sort_key(record: Any) -> tuple:
    if isinstance(record, ResolvedFacePlacement):
        return ("placement", record.placement_sha256, record.alpha)
    if isinstance(record, PlacementRefusal):
        return ("refusal", record.reason, record.detail_sha256,
                record.refusal_sha256)
    raise ValueError("placement sort key requires a placement or refusal record")


def canonical_placement_order(records: Sequence[Any]) -> tuple:
    if not isinstance(records, Sequence):
        raise ValueError("placement records must be a sequence")
    return tuple(sorted(records, key=placement_sort_key))


# ---------------------------------------------------------------------------
# Universe gates and replay
# ---------------------------------------------------------------------------

def _certificate(c2rb_result):
    if (not isinstance(c2rb_result, plane_relative_translation.C2RBResult)
            or c2rb_result.status != "complete"
            or c2rb_result.placement_readiness != "ready"
            or c2rb_result.connected is not True
            or len(c2rb_result.components) != 1):
        raise _InputRefusal("c2rb_gate_not_unique", {"field": "c2rb_result"})
    component = c2rb_result.components[0]
    if (component.outcome != "unique_validated" or component.cycle_rank < 1
            or component.exhaustive is not True or component.failing_chords
            or any(cycle.matching_provenance_count <= 0
                   for cycle in component.cycle_residuals)
            or len(component.translation_classes) != 1):
        raise _InputRefusal("c2rb_gate_not_unique", {"field": "c2rb_result"})
    return component, component.translation_classes[0]


def _check_v2a(v2a_result) -> None:
    if (not isinstance(v2a_result, v2a.V2AEndpointResult)
            or v2a_result.status != "complete"
            or v2a_result.classification != "endpoint_unique"
            or v2a_result.exhaustive is not True
            or v2a_result.lower_bound_only is not False
            or v2a_result.translation_class_count != 1
            or v2a_result.endpoint_assignment_count != 1
            or len(v2a_result.witnesses) != 1
            or v2a_result.refusals
            or v2a_result.budget_failure_reason is not None):
        raise _InputRefusal("endpoint_gate_not_unique", {"field": "v2a_result"})


def _check_v2b(v2b_result) -> tuple:
    """Return the fully determined per-axis alpha vector from one V2-B result.

    Every one of the three world axes must be ``determined``; a partial or
    abstained axis is the typed atomic ``anchor_axis_unresolved`` refusal.
    """
    if (not isinstance(v2b_result, v2b.V2BAnchorResult)
            or v2b_result.status not in ("determined", "partial", "abstained")
            or v2b_result.prediction is None):
        raise _InputRefusal("anchor_gate_not_unique", {"field": "v2b_result"})
    prediction = v2b_result.prediction
    determined: dict[int, tuple[int, int]] = {}
    for axis in prediction.axes:
        if axis.status == "determined":
            if axis.alpha_dyadic is None:
                raise _InputRefusal("anchor_gate_not_unique",
                                    {"field": "v2b_result"})
            determined[axis.world_axis] = tuple(axis.alpha_dyadic)
    missing = [axis for axis in V2_PLACEMENT_WORLD_AXES
               if axis not in determined]
    if missing:
        raise _InputRefusal("anchor_axis_unresolved", {"axes": missing})
    return tuple(
        _reduce(_canonical_dyadic(determined[axis]))
        for axis in V2_PLACEMENT_WORLD_AXES)


def _readings_index(readings: Sequence) -> dict:
    ordered: dict[str, list] = {}
    identities: dict[int, Any] = {}
    for reading in readings:
        if not isinstance(reading, lattice_planes.PlaneReading):
            raise _InputRefusal("invalid_input", {"field": "readings"})
        if id(reading) in identities:
            continue
        identities[id(reading)] = reading
        ordered.setdefault(reading.frame, []).append(reading)
    index: dict[tuple, Any] = {}
    for frame, values in ordered.items():
        for position, reading in enumerate(
                sorted(values, key=lattice_planes.reading_order_key)):
            index[(frame, position, reading.axis)] = reading
    return index


def _replay_universe(universe: PlacementUniverse) -> tuple:
    component, certificate = _certificate(universe.c2rb_result)
    _check_v2a(universe.v2a_result)
    alpha = _check_v2b(universe.v2b_result)
    try:
        if not plane_relative_translation.validate_relative_translation_graph(
                universe.c2rb_result, universe.vector_result,
                universe.axis_support, universe.p1d_result, universe.matches,
                universe.phase_evidence, readings=universe.readings,
                signatures=universe.signatures,
                comparison_tolerance=universe.comparison_tolerance,
                closure_tolerance=universe.closure_tolerance):
            raise _InputRefusal("replay_mismatch", {"stage": "c2rb"})
        if not v2a.validate_endpoint_provenance_result(
                universe.v2a_result, universe.c2rb_result,
                universe.vector_result, universe.axis_support,
                universe.p1d_result, universe.matches, universe.phase_evidence,
                readings=universe.readings, signatures=universe.signatures,
                comparison_tolerance=universe.comparison_tolerance,
                closure_tolerance=universe.closure_tolerance):
            raise _InputRefusal("replay_mismatch", {"stage": "v2a"})
        if not v2b.validate_phase_anchor_replay(
                universe.v2b_result, universe.c2rb_result,
                universe.vector_result, universe.axis_support,
                universe.p1d_result, universe.matches, universe.phase_evidence,
                readings=universe.readings, signatures=universe.signatures,
                comparison_tolerance=universe.comparison_tolerance,
                closure_tolerance=universe.closure_tolerance,
                v2a_result=universe.v2a_result,
                split_sources=universe.split_sources,
                budget=universe.anchor_budget):
            raise _InputRefusal("replay_mismatch", {"stage": "v2b"})
    except _InputRefusal:
        raise
    except (AttributeError, TypeError, ValueError, OverflowError,
            FloatingPointError, ZeroDivisionError) as error:
        raise _InputRefusal("invalid_input",
                            {"detail": type(error).__name__}) from error

    translations = {
        frame: tuple(_signed_dyadic(value) for value in xyz)
        for frame, xyz in certificate.translations_dyadic_xyz}
    supported: dict[tuple[str, int], bool] = {}
    for record in universe.axis_support.frame_support:
        supported[(record.frame, record.world_axis)] = bool(
            record.connected_to_global_anchor and record.refusal is None)
    binding = {
        "c2rb_result_sha256": universe.c2rb_result.result_sha256,
        "component_sha256": component.component_sha256,
        "certificate_sha256": certificate.certificate_sha256,
        "gauge_anchor": certificate.gauge_anchor,
        "v2a_result_sha256": universe.v2a_result.result_sha256,
        "v2b_result_sha256": universe.v2b_result.result_sha256,
        "v2b_status": universe.v2b_result.status,
        "determined_axes": list(V2_PLACEMENT_WORLD_AXES),
        "anchor_digest": placement_anchor_digest(alpha),
        "comparison_tolerance": float(universe.comparison_tolerance),
        "closure_tolerance": float(universe.closure_tolerance),
    }
    return component, certificate, translations, supported, alpha, binding


def _endpoint_assignment(v2a_result) -> dict:
    """Map each reading identity digest to its unique endpoint residue pair."""
    if len(v2a_result.witnesses) != 1:
        raise _InputRefusal("endpoint_gate_not_unique", {"field": "witnesses"})
    labels = v2a_result.witnesses[0].labels
    assignment: dict[str, tuple[int, int]] = {}
    for entry in labels:
        if (not isinstance(entry, tuple) or len(entry) != 4
                or not _hex64(entry[0]) or type(entry[1]) is not int
                or type(entry[2]) is not int or not _hex64(entry[3])):
            raise _InputRefusal("endpoint_gate_not_unique",
                                {"field": "witness.labels"})
        if entry[0] in assignment:
            raise _InputRefusal("endpoint_gate_not_unique",
                                {"field": "witness.labels"})
        assignment[entry[0]] = (entry[1], entry[2])
    if len(assignment) != len(v2a_result.reading_domains):
        raise _InputRefusal("endpoint_gate_not_unique",
                            {"field": "witness.coverage"})
    return assignment


# ---------------------------------------------------------------------------
# Support binding and replay
# ---------------------------------------------------------------------------

def _reading_for_domain(index: Mapping[tuple, Any], domain: Any) -> Any:
    identity = domain.reading_identity
    reading = index.get(
        (identity.frame, identity.canonical_index, identity.reading_axis))
    if reading is None:
        raise _InputRefusal("missing_provenance", {"field": "reading"})
    if reading.axis != identity.reading_axis:
        raise _InputRefusal("provenance_mismatch", {"field": "reading.axis"})
    return reading


def _selected_alternative(reading: Any, selected_b: Any) -> Any:
    try:
        matches = reading.matching_alternatives(selected_b)
    except (TypeError, ValueError, OverflowError):
        matches = ()
    if len(matches) != 1:
        raise _InputRefusal("alternative_not_retained",
                            {"field": "reading.alternatives"})
    return v1.RetainedAlternative.build(matches[0])


def _rebuild_support(
    universe: PlacementUniverse,
    reading: Any,
    identity: Any,
    support: Any,
) -> Any:
    usable = None
    for entry in universe.usable_masks:
        if (not isinstance(entry, tuple) or len(entry) != 2
                or not _hex64(entry[0])):
            raise _InputRefusal("missing_provenance", {"field": "usable_masks"})
        if entry[0] == support.reading_identity_sha256:
            usable = entry[1]
            break
    if usable is None:
        raise _InputRefusal("missing_provenance", {"field": "usable_mask"})
    try:
        rebuilt = v1.build_aligned_region_support(
            reading=reading, reading_identity=identity,
            reading_universe=universe.readings, usable_mask=usable,
            image_sha256=support.image_sha256,
            blocks_per_period=support.alternative.blocks_per_period,
            origin_uv=support.origin_uv, grid_shape=support.grid_shape,
            budget=universe.support_budget,
            phase_anchor_sha256=support.phase_anchor_sha256)
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise _InputRefusal("support_replay_mismatch",
                            {"detail": type(error).__name__}) from error
    if not isinstance(rebuilt, v1.AlignedRegionSupport):
        raise _InputRefusal("support_replay_mismatch",
                            {"reason": getattr(rebuilt, "reason", None)})
    # Compare the *actual canonical records*, not the retained digest alone.  A
    # digest-preserving reseal (a support that keeps the accepted
    # ``support_sha256`` while a cell field, e.g. the last row, is rewritten to
    # 999) replays to the same digest but is a different record; it is refused
    # here and the rebuilt record is the only one ever consumed.
    try:
        replayed_record = rebuilt.to_record()
        claimed_record = support.to_record()
    except (AttributeError, TypeError, ValueError) as error:
        raise _InputRefusal("support_replay_mismatch",
                            {"detail": type(error).__name__}) from error
    if replayed_record != claimed_record:
        raise _InputRefusal("support_replay_mismatch",
                            {"field": "support_record"})
    return rebuilt


def _bind_support(
    universe: PlacementUniverse,
    component: Any,
    index: Mapping[tuple, Any],
    assignment: Mapping[str, tuple[int, int]],
    alpha: tuple,
    support: Any,
) -> tuple:
    """Bind one support to its reading, assignment, evidence and anchor.

    Returns the ``(reading, domain, rebuilt_support)`` triple after every
    provenance cross-link has been replayed and the support has been rebuilt
    byte-for-byte.  The returned support is the canonical rebuilt record; the
    caller consumes only that record, never the supplied object.
    """
    if not isinstance(support, v1.AlignedRegionSupport):
        raise _InputRefusal("invalid_input", {"field": "supports"})
    identity = support.reading_identity
    digest = support.reading_identity_sha256
    owners = [domain for domain in universe.v2a_result.reading_domains
              if domain.reading_identity.identity_sha256 == digest]
    if len(owners) != 1:
        raise _InputRefusal("missing_provenance",
                            {"field": "support.reading_identity"})
    domain = owners[0]
    if domain.reading_identity != identity:
        raise _InputRefusal("provenance_mismatch",
                            {"field": "support.reading_identity"})
    if assignment.get(digest) is None:
        raise _InputRefusal("missing_provenance",
                            {"field": "support.endpoint_assignment"})
    reading = _reading_for_domain(index, domain)

    # Selected retained alternative: exactly the reading's alternative for the
    # domain's selected b, so a 1 -> 2 distance reseal cannot survive.
    selected = _selected_alternative(reading, domain.selected_b)
    if selected != support.alternative:
        raise _InputRefusal("alternative_not_retained",
                            {"field": "support.alternative"})

    # Calibration: rebuild the rectification calibration from the reading and
    # require exact equality with the support's frozen calibration.
    try:
        calibration = v1.RegionSupportCalibration.build(
            axis_matrix=reading.rectification.axis_matrix,
            focal=reading.rectification.focal,
            principal_point=reading.rectification.principal_point,
            centre=reading.rectification.centre, side=reading.side)
    except (AttributeError, TypeError, ValueError) as error:
        raise _InputRefusal("provenance_mismatch",
                            {"field": "support.calibration"}) from error
    if calibration != support.calibration:
        raise _InputRefusal("provenance_mismatch",
                            {"field": "support.calibration"})

    # Original region mask / image evidence.
    evidence = reading.region_evidence
    if not isinstance(evidence, lattice_planes.PlaneRegionEvidence):
        raise _InputRefusal("missing_provenance", {"field": "region_evidence"})
    if (support.source_region_id != evidence.source_region_id
            or support.region_mask_sha256 != evidence.mask_sha256
            or support.image_shape != (int(evidence.image_shape[0]),
                                       int(evidence.image_shape[1]))):
        raise _InputRefusal("provenance_mismatch", {"field": "region_evidence"})
    if (not _hex64(support.usable_mask_sha256)
            or support.usable_mask_sha256 == "0" * 64):
        raise _InputRefusal("missing_provenance", {"field": "usable_mask_sha256"})

    # Accepted phase-anchor digest: a foreign digest is refused outright.
    if support.phase_anchor_sha256 != placement_anchor_digest(alpha):
        raise _InputRefusal("provenance_mismatch",
                            {"field": "support.phase_anchor_sha256"})

    # Rebuild the whole support from the original reading, universe, usable mask
    # and budget; only the canonical rebuilt record is consumed.
    rebuilt = _rebuild_support(
        universe, reading, domain.reading_identity, support)
    return reading, domain, rebuilt


# ---------------------------------------------------------------------------
# Face resolution
# ---------------------------------------------------------------------------

def _resolve_face(
    support: Any,
    domain: Any,
    reading: Any,
    translations: Mapping[str, tuple],
    supported: Mapping[tuple[str, int], bool],
    alpha: tuple,
    counter: _Counter,
) -> ResolvedFace:
    counter.tick("supports_checked")
    translation = translations.get(reading.frame)
    if translation is None:
        raise _InputRefusal("missing_provenance", {"field": "translation"})
    # Every required frame/world axis must be explicitly supported: an
    # unconstrained axis is min-norm-filled with 0.0 by the solver and is never
    # evidence.  The face needs its normal axis and both tangent axes.
    for axis in V2_PLACEMENT_WORLD_AXES:
        counter.tick("frame_axis_lookups")
        if not supported.get((reading.frame, axis), False):
            raise _InputRefusal("translation_unsupported",
                                {"frame": reading.frame, "axis": axis})
    quadrant = int(domain.absolute_quadrant)
    normal_axis, normal_sign = plane_graph.axis_transform(reading.axis, quadrant)
    side = _finite(reading.side)
    if side is None or abs(side) != 1.0:
        raise _InputRefusal("invalid_input", {"field": "reading.side"})
    distance = _finite(support.alternative.distance_blocks)
    if distance is None or distance <= 0.0:
        raise _InputRefusal("alternative_not_retained",
                            {"field": "distance_blocks"})
    counter.tick("integer_checks")
    face = face_axis_coordinate(
        local_label=reading.axis, quadrant=quadrant, c=side, d=distance,
        translation_dyadic=_reduce(translation[normal_axis]),
        alpha_dyadic=alpha[normal_axis])
    if not face["integer"]:
        raise _InputRefusal("face_integer_unresolved", {
            "frame": reading.frame,
            "axis": normal_axis,
            "residual": list(face["residual"]),
            "alpha": list(face["alpha"]),
            "face_gauge": list(face["face_gauge"]),
        })
    # Tangent labels are the sorted axes excluding the normal label, local u
    # first and local v second, each mapped through the shared transform.
    tangent_labels = tuple(label for label in V2_PLACEMENT_WORLD_AXES
                           if label != reading.axis)
    origins = (Fraction(*support.origin_uv[0]), Fraction(*support.origin_uv[1]))
    if (support.column_label, support.row_label) != tangent_labels:
        raise _InputRefusal("provenance_mismatch",
                            {"field": "support.labels"})
    mappings = []
    for local_axis, local_label in zip(V2_PLACEMENT_LOCAL_AXES, tangent_labels):
        origin = origins[0 if local_axis == "u" else 1]
        coordinate = tangent_axis_coordinate(
            local_label=local_label, quadrant=quadrant, origin=origin,
            translation_dyadic=_reduce(translation[
                plane_graph.axis_transform(local_label, quadrant)[0]]),
            alpha_dyadic=alpha[plane_graph.axis_transform(
                local_label, quadrant)[0]])
        counter.tick("anchors")
        counter.tick("integer_checks")
        if not coordinate["integer"]:
            raise _InputRefusal("face_integer_unresolved", {
                "frame": reading.frame,
                "local_axis": local_axis,
                "raw": list(coordinate["raw"]),
                "alpha": list(coordinate["alpha"]),
            })
        mappings.append(_make_tangent_mapping(
            local_axis=local_axis, local_label=int(local_label),
            world_axis=int(coordinate["world_axis"]),
            world_sign=int(coordinate["world_sign"]),
            origin=_reduce(origin), anchor=int(coordinate["anchor"])))
    mapping_u, mapping_v = mappings
    cells = []
    for cell in support.cells:
        if cell.state != "supported":
            continue
        lateral_u = _cell_lower_endpoint(
            mapping_u.anchor, mapping_u.world_sign, int(cell.column))
        lateral_v = _cell_lower_endpoint(
            mapping_v.anchor, mapping_v.world_sign, int(cell.row))
        counter.tick("support_cells")
        counter.tick("anchors", 2)
        cells.append(_make_cell(
            row=int(cell.row), column=int(cell.column),
            lateral_u=lateral_u, lateral_v=lateral_v))
    counter.tick("faces")
    return ResolvedFace.build(
        frame=reading.frame,
        plane_index=int(domain.reading_identity.canonical_index),
        reading_axis=int(reading.axis),
        absolute_quadrant=quadrant,
        source_region_id=support.source_region_id,
        reading_identity_sha256=support.reading_identity_sha256,
        support_sha256=support.support_sha256,
        world_normal_axis=int(normal_axis),
        world_normal_sign=int(normal_sign),
        c=_reduce(Fraction(*side.as_integer_ratio())),
        d=_reduce(Fraction(*distance.as_integer_ratio())),
        translation=tuple(
            _reduce(translation[axis]) for axis in V2_PLACEMENT_WORLD_AXES),
        alpha=tuple(_reduce(_canonical_dyadic(alpha[axis]))
                    for axis in V2_PLACEMENT_WORLD_AXES),
        face_gauge=tuple(face["face_gauge"]),
        face=int(face["face"]),
        outward_sign=int(face["outward_sign"]),
        normal_block=int(face["normal_block"]),
        tangent_mappings=tuple(mappings),
        cells=tuple(cells))


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def resolve_face_placement(
    *,
    universe: PlacementUniverse,
    supports: Sequence["v1.AlignedRegionSupport"],
    budget: PlacementBudget,
    producer_version: str = V2_PLACEMENT_PRODUCER_VERSION,
) -> ResolvedFacePlacement | PlacementRefusal:
    """Resolve one replay-validated mixed-normal universe onto exact integers.

    ``universe`` carries the single accepted C2-R-B/V2-A/V2-B component and the
    original usable masks.  Each support is bound to its unique endpoint
    assignment, selected retained alternative, calibration, original evidence and
    accepted anchor digest, then rebuilt from its original sources; only then are
    its supported cells placed.  Validation completes before any budget
    decision, and exceeding a cap discards every partial face atomically.
    """
    if not isinstance(budget, PlacementBudget):
        return _make_refusal("invalid_input", {"field": "budget"})
    counter = _Counter(budget)
    try:
        if type(producer_version) is not str \
                or producer_version != V2_PLACEMENT_PRODUCER_VERSION:
            raise _InputRefusal("invalid_input", {"field": "producer_version"})
        if not isinstance(universe, PlacementUniverse):
            raise _InputRefusal("invalid_input", {"field": "universe"})
        if (not isinstance(supports, Sequence) or isinstance(supports, (str, bytes))
                or any(not isinstance(support, v1.AlignedRegionSupport)
                       for support in supports)):
            raise _InputRefusal("invalid_input", {"field": "supports"})
        if (not isinstance(universe.readings, tuple) or not universe.readings
                or any(not isinstance(reading, lattice_planes.PlaneReading)
                       for reading in universe.readings)):
            raise _InputRefusal("invalid_input", {"field": "readings"})

        component, certificate, translations, supported, alpha, binding = \
            _replay_universe(universe)
        index = _readings_index(universe.readings)
        assignment = _endpoint_assignment(universe.v2a_result)

        bound = []
        seen = set()
        for support in supports:
            if support.reading_identity_sha256 in seen:
                raise _InputRefusal("provenance_mismatch",
                                    {"field": "duplicate_support"})
            seen.add(support.reading_identity_sha256)
            reading, domain, rebuilt = _bind_support(
                universe, component, index, assignment, alpha, support)
            bound.append((rebuilt, domain, reading))

        faces = []
        for validated_support, domain, reading in bound:
            faces.append(_resolve_face(validated_support, domain, reading,
                                       translations, supported, alpha, counter))
        faces = tuple(sorted(faces, key=face_sort_key))
        work = counter.snapshot()
        return ResolvedFacePlacement.build(
            schema_version=V2_PLACEMENT_SCHEMA,
            producer_version=V2_PLACEMENT_PRODUCER_VERSION,
            gauge_state=V2_PLACEMENT_GAUGE_STATE,
            gauge_axes=V2_PLACEMENT_WORLD_AXES,
            alpha=tuple(_reduce(_canonical_dyadic(alpha[axis]))
                        for axis in V2_PLACEMENT_WORLD_AXES),
            component_bindings=(dict(binding),),
            faces=faces,
            work=work)
    except _BudgetStop as stop:
        return _make_refusal("budget_exhausted", {
            "cap": stop.cap, "limit": stop.limit,
            "observed": stop.observed}, counter.snapshot())
    except _InputRefusal as refusal:
        return _make_refusal(refusal.reason, refusal.detail)
