"""M4 V3 isolated geometry-only supported-plane volume bridge.

This module implements only the isolated V3 step of
``docs/reports/M4-placement-volume-bridge-contract-2026-09-19.md``: turn one
already replay-validated, solver-produced
:class:`~plane_placement.ResolvedFacePlacement` plus its accepted
:class:`~plane_region_support.AlignedRegionSupport` records into a deterministic,
schema-versioned, *geometry-only* set of supported block coordinates, typed
conflicts and explicitly omitted unknown cells.

It defines no block material, no appearance/state fusion, no air or terrain, no
shell/gap filling, no extrusion, no bounding-box fill, no CLI/UI, no manual
metadata, no persistence schema, no litematic wiring and no promotion gate.  It
does **not** import any export, exporter, CLI, UI or project-store module; the
only downstream adapter is the explicit geometry-only ``BlockVolume`` preview
below, whose observations are marked inferred with zero confidence and whose
record states ``exact_state_eligible=False`` so a fixed preview state can never
be consumed as M5 material evidence or exact-state scoring.

Load-bearing rules:

* **One adjacent block per supported face cell.**  For every V1 ``supported``
  cell of a resolved face, exactly one block is emitted at the face's
  ``normal_block`` on the face's world normal axis and the cell's exact integer
  lower endpoints on the face's two tangent world axes.  Nothing is extruded
  behind a face, no bounding box or polygon is filled, and no unsupported cell
  is invented; a cell that V1 classified ``unknown`` is omitted and preserved as
  a typed :class:`OmittedCell`, never as air.
* **Deterministic merge of compatible faces.**  Contributions that land on the
  same coordinate are merged into one :class:`SupportedCellObservation` that
  retains every real frame ID, reading identity, support digest and resolved-face
  digest.  Every accepted claim bounds the merged unit cell, so perpendicular
  corner claims and the two opposite faces of one cube are compatible surface
  evidence and merge with no conflict.  :class:`CellConflict` is preserved but
  reserved for explicitly incompatible evidence: a coordinate is recorded as a
  conflict only when one of its claims does not bound that cell.  The bridge
  never chooses between claims and never ranks them by confidence.
* **Source-bound replay.**  :func:`build_plane_volume` requires the original
  :class:`~plane_placement.PlacementUniverse`, the original V1 supports, the
  candidate :class:`~plane_placement.ResolvedFacePlacement`, and both the
  placement and volume budgets.  It reruns
  :func:`plane_placement.resolve_face_placement` from the original universe,
  supports and placement budget, then compares the canonical placement records
  and every face/support digest before any volume budget decision.  A freshly
  resealed placement that keeps the original bindings (for example a ``+100``
  normal translation) replays its own fields but not the source records, so it is
  refused.  Every support is additionally cross-checked against its original
  reading, region-mask, image and usable-mask evidence.
* **Explicit geometry-only, immutable preview.**
  :func:`to_geometry_only_preview` requires the same source replay context,
  reruns the accepted source replay and derives the full authoritative record
  set, then validates every nested result record and the source replay *before*
  any volume budget decision.  A phantom coordinate with a tiny budget is
  therefore a typed ``replay_mismatch`` rather than ``budget_exhausted``, and
  malformed result data (for example ``cells=(None,)``) becomes a typed refusal
  instead of an escaped ``AttributeError``.  The preview state is copied into an
  immutable name+tuple-properties snapshot, so a list-backed ``BlockState``
  cannot be mutated into the sealed preview or its exported observations.  The
  preview record is immutable and never exposes a mutable ``BlockVolume``: it
  stores an immutable tuple of typed :class:`PreviewBlock` records and produces a
  fresh independent state and ``BlockVolume`` only through
  :meth:`GeometryOnlyPreview.to_block_volume`.  ``exact_state_eligible`` is hard
  ``False``.  A modified result with a phantom coordinate or a stale digest is
  refused before any preview is built.
* **Atomic, truthful budgets.**  Every input, provenance and replay check is
  completed before any budget decision, so a tampered record can never be
  reported as ``budget_exhausted``.  Each work counter is incremented exactly
  when its unit of work is performed; a cap excess discards every partial cell
  and carries only the observed counter.  The preview derives the authoritative
  full record set once with an unlimited recorder, then replays those recorded
  work events through the caller's budget, so the preview's budget refusal is
  byte-identical to the builder's.  The dense bounding-box size is preflighted
  against its own cap before any dense allocation could occur.
* The whole-build integer translation gauge stays explicitly unresolved and is
  carried through unchanged from the placement.  ``complete`` stays ``False``
  until the material and production gates pass.

The module performs no IO and has no filesystem, network, process, environment,
or dynamic-import access.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields as _dataclass_fields
from fractions import Fraction
from typing import Any, Mapping, Sequence

from mc_photogrammetry.domain.models import (
    BlockObservation,
    BlockState,
    BlockVolume,
    ObservationKind,
)
from mc_photogrammetry.reconstruction import plane_placement
from mc_photogrammetry.reconstruction import plane_region_support as v1


__all__ = [
    "V3_VOLUME_SCHEMA",
    "V3_VOLUME_PRODUCER_VERSION",
    "V3_VOLUME_GAUGE_STATE",
    "V3_VOLUME_WORLD_AXES",
    "V3_VOLUME_CONFLICT_KINDS",
    "V3_VOLUME_REFUSAL_REASONS",
    "V3_VOLUME_COMPLETE",
    "V3_VOLUME_EXPORT_BOUNDS",
    "V3_VOLUME_PREVIEW_STATE_CONFIDENCE",
    "PlaneVolumeBudget",
    "VolumeWork",
    "VolumeRefusal",
    "CellClaim",
    "SupportedCellObservation",
    "CellConflict",
    "OmittedCell",
    "PlaneVolumeResult",
    "PreviewBlock",
    "GeometryOnlyPreview",
    "make_volume_budget",
    "volume_sort_key",
    "canonical_volume_order",
    "build_plane_volume",
    "to_geometry_only_preview",
]

V3_VOLUME_SCHEMA = "m4-v3-supported-plane-volume-1"
V3_VOLUME_PRODUCER_VERSION = "v3-supported-volume-bridge-1"
V3_VOLUME_GAUGE_STATE = plane_placement.V2_PLACEMENT_GAUGE_STATE
V3_VOLUME_WORLD_AXES: tuple[int, ...] = (0, 1, 2)
V3_VOLUME_CONFLICT_KINDS: tuple[str, ...] = (
    "opposed_sides",
    "perpendicular_faces",
)
# ``complete`` is deliberately hard-frozen until M5 material fusion and the
# production gates pass; a future acceptance changes the schema/version first.
V3_VOLUME_COMPLETE = False
# The exporter must be called with ``bounds=None`` so normalization is derived
# from the emitted cells alone, never from a supplied/reference bound.
V3_VOLUME_EXPORT_BOUNDS = None
# Geometry-only preview observations are inferred and uncalibrated so they can
# never be consumed as observed exact-state evidence.
V3_VOLUME_PREVIEW_STATE_CONFIDENCE = 0.0
V3_VOLUME_REFUSAL_REASONS: tuple[str, ...] = (
    "invalid_input",
    "missing_provenance",
    "provenance_mismatch",
    "replay_mismatch",
    "placement_replay_refused",
    "support_mismatch",
    "support_replay_mismatch",
    "dense_bounds_exceeded",
    "budget_exhausted",
    "geometry_only_required",
    "preview_state_invalid",
    "air_state_forbidden",
)
#: Preview-state names that would smuggle an air/terrain claim into the volume.
_AIR_STATE_NAMES = frozenset({"air", "cave_air", "void_air"})


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


def _validate_pair(value: Any) -> bool:
    return (isinstance(value, tuple) and len(value) == 2
            and all(type(part) is int for part in value)
            and value[1] > 0 and math.gcd(value[0], value[1]) == 1)


def _validate_coordinate(value: Any) -> bool:
    return (isinstance(value, tuple) and len(value) == 3
            and all(type(part) is int for part in value))


def _pair(value: Fraction) -> tuple[int, int]:
    value = Fraction(value)
    if value == 0:
        return (0, 1)
    return (int(value.numerator), int(value.denominator))


def _record_fields(value: Any, exclude: Sequence[str]) -> dict:
    """All dataclass fields of ``value`` except the given computed digests."""
    excluded = set(exclude)
    return {field.name: getattr(value, field.name)
            for field in _dataclass_fields(value) if field.name not in excluded}


def _cell_lower_endpoint(anchor: int, world_sign: int, index: int) -> int:
    """Lower world interval endpoint of local unit cell ``index``.

    Re-derived independently here (same rule as the resolved face) so a
    tampered face cell cannot survive the V3 replay cross-check.
    """
    if world_sign > 0:
        return anchor + index
    return anchor - (index + 1)


def _is_air_state_name(name: Any) -> bool:
    return type(name) is str and name.rsplit(":", 1)[-1].lower() in _AIR_STATE_NAMES


# ---------------------------------------------------------------------------
# Budget, work and refusal
# ---------------------------------------------------------------------------

_BUDGET_FIELDS: tuple[str, ...] = (
    "max_faces",
    "max_support_cells",
    "max_contributions",
    "max_coordinates",
    "max_conflicts",
    "max_omitted_unknown",
    "max_dense_bounds_cells",
)
_WORK_FIELDS: tuple[str, ...] = (
    "faces",
    "support_cells",
    "contributions",
    "coordinates",
    "conflicts",
    "omitted_unknown",
    "dense_bounds_cells",
)
_WORK_TO_CAP: dict[str, str] = {
    "faces": "max_faces",
    "support_cells": "max_support_cells",
    "contributions": "max_contributions",
    "coordinates": "max_coordinates",
    "conflicts": "max_conflicts",
    "omitted_unknown": "max_omitted_unknown",
    "dense_bounds_cells": "max_dense_bounds_cells",
}
_CAP_TO_WORK = {cap: work for work, cap in _WORK_TO_CAP.items()}
_TICKED_WORK: tuple[str, ...] = tuple(
    name for name in _WORK_FIELDS if name != "dense_bounds_cells")
_BUDGET_LIKE_REASONS = ("budget_exhausted", "dense_bounds_exceeded")


@dataclass(frozen=True, slots=True)
class PlaneVolumeBudget:
    """Explicit positive whole-request limits.  No limit has a default."""

    max_faces: int
    max_support_cells: int
    max_contributions: int
    max_coordinates: int
    max_conflicts: int
    max_omitted_unknown: int
    max_dense_bounds_cells: int
    budget_sha256: str

    def __post_init__(self) -> None:
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 1
               for name in _BUDGET_FIELDS):
            raise ValueError("volume budget limits must be positive integers")
        if (not _hex64(self.budget_sha256)
                or self.budget_sha256 != _sha256(self._payload())):
            raise ValueError("volume budget digest mismatch")

    def _payload(self) -> dict:
        return {name: getattr(self, name) for name in _BUDGET_FIELDS}

    def to_record(self) -> dict:
        record = self._payload()
        record["budget_sha256"] = self.budget_sha256
        return record


def make_volume_budget(**caps: int) -> PlaneVolumeBudget:
    """Build a self-validating volume budget from explicit caps (no defaults)."""
    payload: dict[str, int] = {}
    for name in _BUDGET_FIELDS:
        if name not in caps:
            raise ValueError("volume budget is missing a limit")
        value = caps[name]
        if type(value) is not int or value < 1:
            raise ValueError("volume budget limits must be positive integers")
        payload[name] = value
    return PlaneVolumeBudget(**payload, budget_sha256=_sha256(payload))


@dataclass(frozen=True, slots=True)
class VolumeWork:
    """Exact observed geometry-only volume work counters."""

    faces: int
    support_cells: int
    contributions: int
    coordinates: int
    conflicts: int
    omitted_unknown: int
    dense_bounds_cells: int
    work_sha256: str

    def __post_init__(self) -> None:
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 0
               for name in _WORK_FIELDS):
            raise ValueError("volume work counters must be non-negative integers")
        if self.coordinates > self.contributions:
            raise ValueError("merged coordinates cannot exceed contributions")
        if self.conflicts > self.coordinates:
            raise ValueError("conflicts cannot exceed merged coordinates")
        if self.contributions + self.omitted_unknown > self.support_cells:
            raise ValueError("work state counts exceed the examined support cells")
        if (not _hex64(self.work_sha256)
                or self.work_sha256 != _sha256(self._payload())):
            raise ValueError("volume work digest mismatch")

    def _payload(self) -> dict:
        return {name: getattr(self, name) for name in _WORK_FIELDS}

    def to_record(self) -> dict:
        record = self._payload()
        record["work_sha256"] = self.work_sha256
        return record


def _make_work(counters: Mapping[str, int]) -> VolumeWork:
    payload: dict[str, int] = {}
    for name in _WORK_FIELDS:
        value = counters.get(name, 0)
        if type(value) is not int or value < 0:
            raise ValueError("volume work counters must be non-negative integers")
        payload[name] = value
    return VolumeWork(**payload, work_sha256=_sha256(payload))


@dataclass(frozen=True, slots=True)
class VolumeRefusal:
    """One typed, atomic volume or preview refusal.  It carries no cells."""

    reason: str
    detail: dict
    detail_sha256: str
    work: VolumeWork | None
    refusal_sha256: str

    def __post_init__(self) -> None:
        if self.reason not in V3_VOLUME_REFUSAL_REASONS:
            raise ValueError("volume refusal reason is not in the closed enum")
        normalized = _json_value(self.detail)
        if not isinstance(normalized, dict):
            raise ValueError("volume refusal detail must be a JSON object")
        encoded = _canonical_json(normalized)
        if self.detail_sha256 != hashlib.sha256(encoded.encode("utf-8")).hexdigest():
            raise ValueError("volume refusal detail digest mismatch")
        if self.work is not None and not isinstance(self.work, VolumeWork):
            raise ValueError("volume refusal work is malformed")
        if "cells" in normalized or "volume" in normalized:
            raise ValueError("volume refusal must not embed partial cells")
        if self.reason in _BUDGET_LIKE_REASONS:
            if self.work is None or set(normalized) != {"cap", "limit", "observed"}:
                raise ValueError("volume budget refusal is malformed")
            cap = normalized["cap"]
            limit = normalized["limit"]
            observed = normalized["observed"]
            if (cap not in _CAP_TO_WORK or type(limit) is not int
                    or type(observed) is not int or limit < 1
                    or observed <= limit):
                raise ValueError("volume budget refusal evidence is malformed")
            if getattr(self.work, _CAP_TO_WORK[cap]) != observed:
                raise ValueError(
                    "volume budget refusal disagrees with its work counter")
        elif self.work is not None:
            raise ValueError("non-budget volume refusal must not carry work")
        payload = {
            "reason": self.reason,
            "detail_sha256": self.detail_sha256,
            "work": None if self.work is None else self.work.to_record(),
        }
        if (not _hex64(self.refusal_sha256)
                or self.refusal_sha256 != _sha256(payload)):
            raise ValueError("volume refusal digest mismatch")
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
                  work: VolumeWork | None = None) -> VolumeRefusal:
    if reason not in V3_VOLUME_REFUSAL_REASONS:
        reason = "invalid_input"
    normalized = _json_value({} if detail is None else detail)
    detail_sha = hashlib.sha256(
        _canonical_json(normalized).encode("utf-8")).hexdigest()
    payload = {
        "reason": reason,
        "detail_sha256": detail_sha,
        "work": None if work is None else work.to_record(),
    }
    return VolumeRefusal(reason, normalized, detail_sha, work, _sha256(payload))


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
    """Truthful, checked volume work counters.

    ``tick`` increments exactly one counter and then checks every ticked cap
    against the observed counters, so a stop always reports the exact observed
    value of the cap that was exceeded.  The dense bounding-box preflight is
    separate and also exact.
    """

    __slots__ = ("budget", "faces", "support_cells", "contributions",
                 "coordinates", "conflicts", "omitted_unknown",
                 "dense_bounds_cells")

    def __init__(self, budget: PlaneVolumeBudget):
        self.budget = budget
        self.faces = 0
        self.support_cells = 0
        self.contributions = 0
        self.coordinates = 0
        self.conflicts = 0
        self.omitted_unknown = 0
        self.dense_bounds_cells = 0

    def tick(self, field: str, amount: int = 1) -> None:
        if field not in _TICKED_WORK or type(amount) is not int or amount < 1:
            raise ValueError("invalid volume counter update")
        setattr(self, field, getattr(self, field) + amount)
        for work_name in _TICKED_WORK:
            observed = getattr(self, work_name)
            limit = getattr(self.budget, _WORK_TO_CAP[work_name])
            if observed > limit:
                raise _BudgetStop(_WORK_TO_CAP[work_name], limit, observed)

    def set_dense_bounds(self, observed: int) -> None:
        if type(observed) is not int or observed < 0:
            raise ValueError("invalid dense bounds count")
        self.dense_bounds_cells = observed
        limit = self.budget.max_dense_bounds_cells
        if observed > limit:
            raise _BudgetStop("max_dense_bounds_cells", limit, observed)

    def snapshot(self) -> VolumeWork:
        return _make_work({name: getattr(self, name) for name in _WORK_FIELDS})


class _RecordingCounter:
    """Unlimited work recorder for the same tick order the budget counter uses.

    It performs the identical ``tick``/``set_dense_bounds`` calls as
    :class:`_Counter` but never stops, so the authoritative full record set can be
    derived once and the caller's volume budget can be applied afterwards by
    replaying the recorded events through a real :class:`_Counter`.  This is what
    lets :func:`to_geometry_only_preview` validate the supplied result and source
    replay *before* any volume budget decision.
    """

    __slots__ = ("events", "counts")

    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.counts: dict[str, int] = {name: 0 for name in _WORK_FIELDS}

    def tick(self, field: str, amount: int = 1) -> None:
        if field not in _TICKED_WORK or type(amount) is not int or amount < 1:
            raise ValueError("invalid volume counter update")
        self.events.append(("tick", field, amount))
        self.counts[field] += amount

    def set_dense_bounds(self, observed: int) -> None:
        if type(observed) is not int or observed < 0:
            raise ValueError("invalid dense bounds count")
        self.events.append(("dense", observed))
        self.counts["dense_bounds_cells"] = observed


# ---------------------------------------------------------------------------
# Geometry-only claim, merged cell, conflict and omission records
# ---------------------------------------------------------------------------

def _claim_payload(value: "CellClaim") -> dict:
    return {
        "frame": value.frame,
        "plane_index": value.plane_index,
        "reading_axis": value.reading_axis,
        "absolute_quadrant": value.absolute_quadrant,
        "source_region_id": value.source_region_id,
        "reading_identity_sha256": value.reading_identity_sha256,
        "support_sha256": value.support_sha256,
        "face_sha256": value.face_sha256,
        "row": value.row,
        "column": value.column,
        "world_normal_axis": value.world_normal_axis,
        "world_normal_sign": value.world_normal_sign,
        "outward_sign": value.outward_sign,
        "face": value.face,
        "normal_block": value.normal_block,
        "tangent_u_axis": value.tangent_u_axis,
        "lateral_u": value.lateral_u,
        "tangent_v_axis": value.tangent_v_axis,
        "lateral_v": value.lateral_v,
        "coordinate": list(value.coordinate),
    }


@dataclass(frozen=True, slots=True)
class CellClaim:
    """One face cell's geometry-only claim on one block coordinate.

    The record preserves real frame/reading/face provenance, the outward sign,
    the exact integer face and normal block, and the two tangent world axes with
    their exact (possibly negative) lower cell endpoints.
    """

    frame: str
    plane_index: int
    reading_axis: int
    absolute_quadrant: int
    source_region_id: str
    reading_identity_sha256: str
    support_sha256: str
    face_sha256: str
    row: int
    column: int
    world_normal_axis: int
    world_normal_sign: int
    outward_sign: int
    face: int
    normal_block: int
    tangent_u_axis: int
    lateral_u: int
    tangent_v_axis: int
    lateral_v: int
    coordinate: tuple[int, int, int]
    claim_sha256: str

    def __post_init__(self) -> None:
        if type(self.frame) is not str or not self.frame:
            raise ValueError("claim frame is invalid")
        if type(self.plane_index) is not int or self.plane_index < 0:
            raise ValueError("claim plane index is invalid")
        if type(self.reading_axis) is not int or self.reading_axis not in (0, 1, 2):
            raise ValueError("claim reading axis is invalid")
        if type(self.absolute_quadrant) is not int \
                or self.absolute_quadrant not in (0, 1, 2, 3):
            raise ValueError("claim quadrant is invalid")
        if (not _hex64(self.source_region_id)
                or not _hex64(self.reading_identity_sha256)
                or not _hex64(self.support_sha256)
                or not _hex64(self.face_sha256)):
            raise ValueError("claim provenance digests are malformed")
        if (type(self.row) is not int or self.row < 0
                or type(self.column) is not int or self.column < 0):
            raise ValueError("claim cell indices are invalid")
        if self.world_normal_axis not in (0, 1, 2):
            raise ValueError("claim world normal axis is invalid")
        if self.world_normal_sign not in (-1, 1) or self.outward_sign not in (-1, 1):
            raise ValueError("claim signs are invalid")
        if type(self.face) is not int or type(self.normal_block) is not int:
            raise ValueError("claim integer coordinates are invalid")
        axes = (self.world_normal_axis, self.tangent_u_axis, self.tangent_v_axis)
        if (any(type(axis) is not int or axis not in (0, 1, 2) for axis in axes)
                or set(axes) != set(V3_VOLUME_WORLD_AXES)):
            raise ValueError("claim axes are not the three world axes")
        if type(self.lateral_u) is not int or type(self.lateral_v) is not int:
            raise ValueError("claim lateral endpoints must be exact integers")
        if not _validate_coordinate(self.coordinate):
            raise ValueError("claim coordinate must be three integers")
        if (self.coordinate[self.world_normal_axis] != self.normal_block
                or self.coordinate[self.tangent_u_axis] != self.lateral_u
                or self.coordinate[self.tangent_v_axis] != self.lateral_v):
            raise ValueError("claim coordinate disagrees with its axes")
        expected_block = self.face - 1 if self.outward_sign > 0 else self.face
        if self.normal_block != expected_block:
            raise ValueError("claim normal block disagrees with its face")
        if (not _hex64(self.claim_sha256)
                or self.claim_sha256 != _sha256(_claim_payload(self))):
            raise ValueError("claim digest mismatch")

    @classmethod
    def build(cls, **fields: Any) -> "CellClaim":
        provisional = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "claim_sha256", "")
        object.__setattr__(provisional, "claim_sha256",
                           _sha256(_claim_payload(provisional)))
        cls.__post_init__(provisional)
        return provisional

    def to_record(self) -> dict:
        record = _claim_payload(self)
        record["claim_sha256"] = self.claim_sha256
        return record


def _claim_sort_key(claim: CellClaim) -> tuple:
    return (claim.frame, claim.plane_index, claim.reading_axis, claim.row,
            claim.column, claim.reading_identity_sha256, claim.support_sha256,
            claim.claim_sha256)


def _claim_bounds_cell(claim: CellClaim, coordinate: tuple[int, int, int]) -> bool:
    """True iff the claim's surface is a boundary face of ``coordinate``'s cell.

    A claim is *compatible* with a merged coordinate exactly when its surface
    bounds that unit cell: the cell lies on the material side of the claimed
    outward normal and the claimed face plane is the cell's own boundary on the
    claim's world normal axis.  Perpendicular corner claims and the two opposite
    faces of one cube all satisfy this predicate, so they merge without conflict;
    only a claim that does not bound the cell is explicitly incompatible.
    """
    if claim.coordinate != coordinate:
        return False
    axis = claim.world_normal_axis
    surface = coordinate[axis] + 1 if claim.outward_sign > 0 else coordinate[axis]
    return claim.face == surface and claim.normal_block == coordinate[axis]


def _incompatible_evidence_kind(
        coordinate: tuple[int, int, int],
        claims: Sequence[CellClaim]) -> str | None:
    """Typed relation of one coordinate's merged claims.

    ``None`` means every claim bounds the same unit cell (compatible evidence).
    A kind is returned only for *explicitly incompatible* evidence -- a claim
    whose surface does not bound the merged cell -- and is named
    ``perpendicular_faces`` when the claims span more than one world normal axis
    and ``opposed_sides`` otherwise.  The bridge never decides between divergent
    claims; it only names a genuinely incompatible relation.
    """
    if not any(not _claim_bounds_cell(claim, coordinate) for claim in claims):
        return None
    if len({claim.world_normal_axis for claim in claims}) > 1:
        return "perpendicular_faces"
    return "opposed_sides"


def _observation_payload(value: "SupportedCellObservation") -> dict:
    return {
        "coordinate": list(value.coordinate),
        "claims": [claim.to_record() for claim in value.claims],
        "frame_ids": list(value.frame_ids),
        "conflict_kind": value.conflict_kind,
    }


@dataclass(frozen=True, slots=True)
class SupportedCellObservation:
    """One emitted block coordinate with every merged geometry-only claim."""

    coordinate: tuple[int, int, int]
    claims: tuple[CellClaim, ...]
    frame_ids: tuple[str, ...]
    conflict_kind: str | None
    observation_sha256: str

    def __post_init__(self) -> None:
        if not _validate_coordinate(self.coordinate):
            raise ValueError("supported cell coordinate must be three integers")
        if (not isinstance(self.claims, tuple) or not self.claims
                or any(not isinstance(claim, CellClaim) for claim in self.claims)):
            raise ValueError("supported cell claims are malformed")
        if tuple(sorted(self.claims, key=_claim_sort_key)) != self.claims:
            raise ValueError("supported cell claims are not in canonical order")
        if any(claim.coordinate != self.coordinate for claim in self.claims):
            raise ValueError("supported cell claims disagree on their coordinate")
        for claim in self.claims:
            if claim.claim_sha256 != _sha256(_claim_payload(claim)):
                raise ValueError("supported cell claim digest does not replay")
        expected_frames = tuple(sorted({claim.frame for claim in self.claims}))
        if self.frame_ids != expected_frames:
            raise ValueError("supported cell frame IDs disagree with its claims")
        expected_kind = _incompatible_evidence_kind(self.coordinate, self.claims)
        if self.conflict_kind != expected_kind:
            raise ValueError("supported cell conflict kind disagrees with its claims")
        if (not _hex64(self.observation_sha256)
                or self.observation_sha256 != _sha256(_observation_payload(self))):
            raise ValueError("supported cell digest mismatch")

    @classmethod
    def build(cls, **fields: Any) -> "SupportedCellObservation":
        provisional = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "observation_sha256", "")
        object.__setattr__(provisional, "observation_sha256",
                           _sha256(_observation_payload(provisional)))
        cls.__post_init__(provisional)
        return provisional

    def to_record(self) -> dict:
        record = _observation_payload(self)
        record["observation_sha256"] = self.observation_sha256
        return record


def _conflict_payload(value: "CellConflict") -> dict:
    return {
        "coordinate": list(value.coordinate),
        "kind": value.kind,
        "claims": [claim.to_record() for claim in value.claims],
        "frame_ids": list(value.frame_ids),
    }


@dataclass(frozen=True, slots=True)
class CellConflict:
    """One coordinate with *explicitly incompatible* merged surface evidence.

    The type is preserved for evidence that genuinely cannot describe one unit
    cell, but it is reserved: compatible perpendicular corner claims and the two
    opposite faces of one cube merge without a conflict, and the constructor
    refuses to record a conflict when every claim bounds the merged cell.
    """

    coordinate: tuple[int, int, int]
    kind: str
    claims: tuple[CellClaim, ...]
    frame_ids: tuple[str, ...]
    conflict_sha256: str

    def __post_init__(self) -> None:
        if not _validate_coordinate(self.coordinate):
            raise ValueError("conflict coordinate must be three integers")
        if self.kind not in V3_VOLUME_CONFLICT_KINDS:
            raise ValueError("conflict kind is not in the closed enum")
        if (not isinstance(self.claims, tuple) or len(self.claims) < 2
                or any(not isinstance(claim, CellClaim) for claim in self.claims)):
            raise ValueError("conflict claims are malformed")
        if tuple(sorted(self.claims, key=_claim_sort_key)) != self.claims:
            raise ValueError("conflict claims are not in canonical order")
        if any(claim.coordinate != self.coordinate for claim in self.claims):
            raise ValueError("conflict claims disagree on their coordinate")
        for claim in self.claims:
            if claim.claim_sha256 != _sha256(_claim_payload(claim)):
                raise ValueError("conflict claim digest does not replay")
        kind = _incompatible_evidence_kind(self.coordinate, self.claims)
        if kind is None:
            raise ValueError(
                "CellConflict is reserved for explicitly incompatible evidence")
        if kind != self.kind:
            raise ValueError("conflict kind disagrees with its claims")
        expected_frames = tuple(sorted({claim.frame for claim in self.claims}))
        if self.frame_ids != expected_frames:
            raise ValueError("conflict frame IDs disagree with its claims")
        if (not _hex64(self.conflict_sha256)
                or self.conflict_sha256 != _sha256(_conflict_payload(self))):
            raise ValueError("conflict digest mismatch")

    @classmethod
    def build(cls, **fields: Any) -> "CellConflict":
        provisional = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "conflict_sha256", "")
        object.__setattr__(provisional, "conflict_sha256",
                           _sha256(_conflict_payload(provisional)))
        cls.__post_init__(provisional)
        return provisional

    def to_record(self) -> dict:
        record = _conflict_payload(self)
        record["conflict_sha256"] = self.conflict_sha256
        return record


def _omitted_payload(value: "OmittedCell") -> dict:
    return {
        "frame": value.frame,
        "plane_index": value.plane_index,
        "reading_axis": value.reading_axis,
        "row": value.row,
        "column": value.column,
        "state": value.state,
        "evidence": value.evidence,
        "source_region_id": value.source_region_id,
        "reading_identity_sha256": value.reading_identity_sha256,
        "support_sha256": value.support_sha256,
        "face_sha256": value.face_sha256,
    }


@dataclass(frozen=True, slots=True)
class OmittedCell:
    """One V1 ``unknown`` cell that was deliberately not emitted.

    The record is the typed omission evidence: no air is created in its place
    and the cell is never counted as supported.
    """

    frame: str
    plane_index: int
    reading_axis: int
    row: int
    column: int
    state: str
    evidence: str
    source_region_id: str
    reading_identity_sha256: str
    support_sha256: str
    face_sha256: str
    omitted_sha256: str

    def __post_init__(self) -> None:
        if type(self.frame) is not str or not self.frame:
            raise ValueError("omitted cell frame is invalid")
        if type(self.plane_index) is not int or self.plane_index < 0:
            raise ValueError("omitted cell plane index is invalid")
        if type(self.reading_axis) is not int or self.reading_axis not in (0, 1, 2):
            raise ValueError("omitted cell reading axis is invalid")
        if (type(self.row) is not int or self.row < 0
                or type(self.column) is not int or self.column < 0):
            raise ValueError("omitted cell indices are invalid")
        if self.state != "unknown":
            raise ValueError("only unknown support cells may be omitted")
        if self.evidence not in v1.V1_SUPPORT_CELL_EVIDENCE \
                or self.evidence == "full":
            raise ValueError("omitted cell evidence is not an unknown reason")
        if (not _hex64(self.source_region_id)
                or not _hex64(self.reading_identity_sha256)
                or not _hex64(self.support_sha256)
                or not _hex64(self.face_sha256)):
            raise ValueError("omitted cell provenance digests are malformed")
        if (not _hex64(self.omitted_sha256)
                or self.omitted_sha256 != _sha256(_omitted_payload(self))):
            raise ValueError("omitted cell digest mismatch")

    @classmethod
    def build(cls, **fields: Any) -> "OmittedCell":
        provisional = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "omitted_sha256", "")
        object.__setattr__(provisional, "omitted_sha256",
                           _sha256(_omitted_payload(provisional)))
        cls.__post_init__(provisional)
        return provisional

    def to_record(self) -> dict:
        record = _omitted_payload(self)
        record["omitted_sha256"] = self.omitted_sha256
        return record


def _omitted_sort_key(record: OmittedCell) -> tuple:
    return (record.frame, record.plane_index, record.reading_axis, record.row,
            record.column, record.reading_identity_sha256)


def _binding_sort_key(binding: Mapping[str, Any]) -> tuple:
    return (binding["frame"], binding["plane_index"], binding["reading_axis"],
            binding["reading_identity_sha256"], binding["support_sha256"])


# ---------------------------------------------------------------------------
# Canonical volume result
# ---------------------------------------------------------------------------

def _volume_payload(value: "PlaneVolumeResult") -> dict:
    return {
        "schema_version": value.schema_version,
        "producer_version": value.producer_version,
        "gauge_state": value.gauge_state,
        "gauge_axes": list(value.gauge_axes),
        "alpha": [list(item) for item in value.alpha],
        "placement_sha256": value.placement_sha256,
        "support_bindings": [dict(binding)
                             for binding in value.support_bindings],
        "cells": [cell.to_record() for cell in value.cells],
        "conflicts": [conflict.to_record() for conflict in value.conflicts],
        "omitted": [record.to_record() for record in value.omitted],
        "cell_bounds": None if value.cell_bounds is None else [
            list(value.cell_bounds[0]), list(value.cell_bounds[1])],
        "dense_bounds_cells": value.dense_bounds_cells,
        "normalization_offset": (None if value.normalization_offset is None
                                 else list(value.normalization_offset)),
        "export_bounds": value.export_bounds,
        "complete": value.complete,
        "work": value.work.to_record(),
    }


def _derived_work(result: "PlaneVolumeResult") -> dict:
    return {
        "faces": len(result.support_bindings),
        "support_cells": sum(int(binding["support_cells"])
                             for binding in result.support_bindings),
        "contributions": sum(len(cell.claims) for cell in result.cells),
        "coordinates": len(result.cells),
        "conflicts": len(result.conflicts),
        "omitted_unknown": len(result.omitted),
        "dense_bounds_cells": result.dense_bounds_cells,
    }


@dataclass(frozen=True, slots=True)
class PlaneVolumeResult:
    """The frozen, versioned, geometry-only supported-plane volume result."""

    schema_version: str
    producer_version: str
    gauge_state: str
    gauge_axes: tuple[int, ...]
    alpha: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    placement_sha256: str
    support_bindings: tuple[dict, ...]
    cells: tuple[SupportedCellObservation, ...]
    conflicts: tuple[CellConflict, ...]
    omitted: tuple[OmittedCell, ...]
    cell_bounds: tuple[tuple[int, int, int], tuple[int, int, int]] | None
    dense_bounds_cells: int
    normalization_offset: tuple[int, int, int] | None
    export_bounds: None
    complete: bool
    work: VolumeWork
    volume_sha256: str

    def __post_init__(self) -> None:
        if (self.schema_version != V3_VOLUME_SCHEMA
                or self.producer_version != V3_VOLUME_PRODUCER_VERSION
                or self.gauge_state != V3_VOLUME_GAUGE_STATE):
            raise ValueError("volume record version or gauge is unsupported")
        if (not isinstance(self.gauge_axes, tuple)
                or self.gauge_axes != V3_VOLUME_WORLD_AXES):
            raise ValueError("volume gauge axes are invalid")
        if (not isinstance(self.alpha, tuple) or len(self.alpha) != 3
                or any(not _validate_pair(item) for item in self.alpha)):
            raise ValueError("volume alpha vector is malformed")
        if not _hex64(self.placement_sha256):
            raise ValueError("volume placement binding is malformed")
        if (not isinstance(self.support_bindings, tuple)
                or not self.support_bindings
                or any(not isinstance(binding, dict)
                       for binding in self.support_bindings)):
            raise ValueError("volume support bindings are malformed")
        if tuple(sorted(self.support_bindings, key=_binding_sort_key)) \
                != self.support_bindings:
            raise ValueError("volume support bindings are not in canonical order")
        binding_keys = [(binding["reading_identity_sha256"],
                         binding["support_sha256"])
                        for binding in self.support_bindings]
        if len(set(binding_keys)) != len(binding_keys):
            raise ValueError("volume support bindings are not unique")
        for binding in self.support_bindings:
            if set(binding) != {
                    "frame", "plane_index", "reading_axis",
                    "reading_identity_sha256", "support_sha256", "face_sha256",
                    "source_region_id", "alternative_distance", "phase_anchor_sha256",
                    "support_cells", "supported_cells", "unknown_cells"}:
                raise ValueError("volume support binding fields are malformed")
            if any(type(binding[name]) is not int or binding[name] < 0
                   for name in ("support_cells", "supported_cells", "unknown_cells")):
                raise ValueError("volume support binding counts are malformed")
            if binding["supported_cells"] + binding["unknown_cells"] \
                    != binding["support_cells"]:
                raise ValueError("volume support binding counts are inconsistent")

        if (not isinstance(self.cells, tuple)
                or any(not isinstance(cell, SupportedCellObservation)
                       for cell in self.cells)):
            raise ValueError("volume cells are malformed")
        coordinates = [cell.coordinate for cell in self.cells]
        if coordinates != sorted(coordinates) or len(set(coordinates)) != len(coordinates):
            raise ValueError("volume cells are not unique and canonically ordered")
        if (not isinstance(self.conflicts, tuple)
                or any(not isinstance(conflict, CellConflict)
                       for conflict in self.conflicts)):
            raise ValueError("volume conflicts are malformed")
        conflict_coordinates = [conflict.coordinate for conflict in self.conflicts]
        if (conflict_coordinates != sorted(conflict_coordinates)
                or len(set(conflict_coordinates)) != len(conflict_coordinates)):
            raise ValueError("volume conflicts are not unique and canonically ordered")
        if (not isinstance(self.omitted, tuple)
                or any(not isinstance(record, OmittedCell)
                       for record in self.omitted)):
            raise ValueError("volume omissions are malformed")
        omitted_keys = [_omitted_sort_key(record) for record in self.omitted]
        if omitted_keys != sorted(omitted_keys):
            raise ValueError("volume omissions are not in canonical order")
        if len(set(omitted_keys)) != len(omitted_keys):
            raise ValueError("volume omissions are not unique")
        # Nested records are re-verified against their own payloads so a cloned
        # record whose stored digest no longer replays cannot ride along inside a
        # freshly resealed result.
        for cell in self.cells:
            if cell.observation_sha256 != _sha256(_observation_payload(cell)):
                raise ValueError("volume cell digest does not replay")
        for conflict in self.conflicts:
            if conflict.conflict_sha256 != _sha256(_conflict_payload(conflict)):
                raise ValueError("volume conflict digest does not replay")
        for record in self.omitted:
            if record.omitted_sha256 != _sha256(_omitted_payload(record)):
                raise ValueError("volume omission digest does not replay")

        cells_by_coordinate = {cell.coordinate: cell for cell in self.cells}
        flagged = {cell.coordinate: cell.conflict_kind for cell in self.cells
                   if cell.conflict_kind is not None}
        if {conflict.coordinate for conflict in self.conflicts} != set(flagged):
            raise ValueError("volume conflicts disagree with its cells")
        for conflict in self.conflicts:
            cell = cells_by_coordinate[conflict.coordinate]
            if (flagged[conflict.coordinate] != conflict.kind
                    or tuple(claim.claim_sha256 for claim in conflict.claims)
                    != tuple(claim.claim_sha256 for claim in cell.claims)):
                raise ValueError("volume conflict disagrees with its merged cell")

        support_cells = sum(int(binding["support_cells"])
                            for binding in self.support_bindings)
        supported_cells = sum(int(binding["supported_cells"])
                              for binding in self.support_bindings)
        unknown_cells = sum(int(binding["unknown_cells"])
                            for binding in self.support_bindings)
        if supported_cells != sum(len(cell.claims) for cell in self.cells):
            raise ValueError("volume binding support counts disagree with its cells")
        if unknown_cells != len(self.omitted):
            raise ValueError("volume binding omissions disagree with its records")
        if supported_cells + unknown_cells != support_cells:
            raise ValueError("volume binding cell counts are inconsistent")

        if self.cells:
            low = tuple(min(cell.coordinate[axis] for cell in self.cells)
                        for axis in V3_VOLUME_WORLD_AXES)
            high = tuple(max(cell.coordinate[axis] for cell in self.cells)
                         for axis in V3_VOLUME_WORLD_AXES)
            expected_bounds = (low, high)
            expected_dense = 1
            for axis in V3_VOLUME_WORLD_AXES:
                expected_dense *= high[axis] - low[axis] + 1
        else:
            expected_bounds = None
            expected_dense = 0
        if self.cell_bounds != expected_bounds:
            raise ValueError("volume bounds are not derived from its emitted cells")
        if self.dense_bounds_cells != expected_dense:
            raise ValueError("volume dense bounds disagree with its emitted cells")
        expected_offset = None if expected_bounds is None else expected_bounds[0]
        if self.normalization_offset != expected_offset:
            raise ValueError("volume normalization offset is not its cell minimum")
        if self.export_bounds is not V3_VOLUME_EXPORT_BOUNDS:
            raise ValueError("volume export must use bounds=None")
        if self.complete is not V3_VOLUME_COMPLETE:
            raise ValueError("volume completeness is frozen at False")
        if not isinstance(self.work, VolumeWork):
            raise ValueError("volume work record is malformed")
        derived = _derived_work(self)
        for name in _WORK_FIELDS:
            if getattr(self.work, name) != derived[name]:
                raise ValueError("volume work does not replay its records")
        if (not _hex64(self.volume_sha256)
                or self.volume_sha256 != _sha256(_volume_payload(self))):
            raise ValueError("volume digest mismatch")

    @classmethod
    def build(cls, **fields: Any) -> "PlaneVolumeResult":
        provisional = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "volume_sha256", "")
        object.__setattr__(provisional, "volume_sha256",
                           _sha256(_volume_payload(provisional)))
        cls.__post_init__(provisional)
        return provisional

    def to_record(self) -> dict:
        record = _volume_payload(self)
        record["volume_sha256"] = self.volume_sha256
        return record


def volume_sort_key(record: Any) -> tuple:
    if isinstance(record, PlaneVolumeResult):
        return ("volume", record.volume_sha256, record.placement_sha256)
    if isinstance(record, VolumeRefusal):
        return ("refusal", record.reason, record.detail_sha256,
                record.refusal_sha256)
    raise ValueError("volume sort key requires a volume or refusal record")


def canonical_volume_order(records: Sequence[Any]) -> tuple:
    if not isinstance(records, Sequence):
        raise ValueError("volume records must be a sequence")
    return tuple(sorted(records, key=volume_sort_key))


# ---------------------------------------------------------------------------
# Source-bound replay and binding
# ---------------------------------------------------------------------------

def _replay_source_placement(
    universe: plane_placement.PlacementUniverse,
    supports: tuple[v1.AlignedRegionSupport, ...],
    placement_budget: plane_placement.PlacementBudget,
) -> plane_placement.ResolvedFacePlacement:
    """Rerun the accepted V2 producer from the original universe and supports.

    The candidate placement is never trusted: the accepted producer is rerun
    from the original :class:`~plane_placement.PlacementUniverse`, the original
    V1 supports and the placement budget, so a self-consistent reseal cannot be
    consumed.  A producer refusal becomes a typed ``placement_replay_refused``
    refusal before any volume budget decision.
    """
    try:
        source = plane_placement.resolve_face_placement(
            universe=universe, supports=supports, budget=placement_budget)
    except (AttributeError, TypeError, ValueError, OverflowError,
            ZeroDivisionError) as error:
        raise _InputRefusal("placement_replay_refused",
                            {"detail": type(error).__name__}) from error
    if not isinstance(source, plane_placement.ResolvedFacePlacement):
        raise _InputRefusal("placement_replay_refused", {
            "reason": getattr(source, "reason", None),
            "detail": getattr(source, "detail", None)})
    return source


def _readings_by_frame_axis(readings: Sequence[Any]) -> dict[tuple[str, int], Any]:
    index: dict[tuple[str, int], Any] = {}
    for reading in readings:
        key = (reading.frame, int(reading.axis))
        if key in index and index[key] is not reading:
            raise _InputRefusal("provenance_mismatch",
                                {"field": "reading_identity"})
        index[key] = reading
    return index


def _cross_check_sources(
    universe: plane_placement.PlacementUniverse,
    supports: tuple[v1.AlignedRegionSupport, ...],
    source_placement: plane_placement.ResolvedFacePlacement,
    candidate_placement: plane_placement.ResolvedFacePlacement,
) -> None:
    """Cross-check every candidate face and support against original records.

    The rerun already rebuilt the whole placement from the original universe and
    supports; this compares the canonical placement records and digests and then
    re-binds every candidate face and support to its source-derived counterpart
    and to the original reading/region/usable-mask evidence.  It runs before any
    volume budget decision.
    """
    supports_by_digest: dict[str, v1.AlignedRegionSupport] = {}
    for support in supports:
        if support.support_sha256 in supports_by_digest:
            raise _InputRefusal("provenance_mismatch",
                                {"field": "duplicate_support"})
        supports_by_digest[support.support_sha256] = support
    candidate_support_ids = {face.support_sha256
                             for face in candidate_placement.faces}
    if candidate_support_ids - set(supports_by_digest):
        raise _InputRefusal("missing_provenance", {"field": "support"})
    if set(supports_by_digest) - candidate_support_ids:
        raise _InputRefusal("provenance_mismatch",
                            {"field": "unreferenced_support"})

    if source_placement.placement_sha256 != candidate_placement.placement_sha256:
        raise _InputRefusal("replay_mismatch", {"field": "placement_sha256"})
    if source_placement.to_record() != candidate_placement.to_record():
        raise _InputRefusal("replay_mismatch", {"field": "placement_record"})

    source_faces: dict[tuple, Any] = {}
    for face in source_placement.faces:
        key = plane_placement.face_sort_key(face)
        if key in source_faces:
            raise _InputRefusal("replay_mismatch", {"field": "source_faces"})
        source_faces[key] = face
    if (len(source_faces) != len(source_placement.faces)
            or len(candidate_placement.faces) != len(source_faces)):
        raise _InputRefusal("replay_mismatch", {"field": "placement_faces"})

    anchor_digest = plane_placement.placement_anchor_digest(source_placement.alpha)

    readings = _readings_by_frame_axis(universe.readings)
    mask_ids = {entry[0] for entry in universe.usable_masks}
    domains = universe.v2a_result.reading_domains
    seen_faces: set[tuple] = set()
    for face in candidate_placement.faces:
        key = plane_placement.face_sort_key(face)
        if key in seen_faces:
            raise _InputRefusal("provenance_mismatch",
                                {"field": "duplicate_face"})
        seen_faces.add(key)
        source_face = source_faces.get(key)
        if source_face is None:
            raise _InputRefusal("missing_provenance", {"field": "face"})
        if (face.face_sha256 != source_face.face_sha256
                or face.to_record() != source_face.to_record()):
            raise _InputRefusal("replay_mismatch", {"field": "face"})
        support = supports_by_digest.get(face.support_sha256)
        if support is None:
            raise _InputRefusal("missing_provenance", {"field": "support"})
        if support.support_sha256 != source_face.support_sha256:
            raise _InputRefusal("provenance_mismatch",
                                {"field": "support_sha256"})
        if support.reading_identity_sha256 != face.reading_identity_sha256:
            raise _InputRefusal("provenance_mismatch",
                                {"field": "reading_identity"})
        if support.phase_anchor_sha256 != anchor_digest:
            raise _InputRefusal("provenance_mismatch",
                                {"field": "phase_anchor_sha256"})
        owners = [domain for domain in domains
                  if domain.reading_identity.identity_sha256
                  == support.reading_identity_sha256]
        if len(owners) != 1:
            raise _InputRefusal("missing_provenance",
                                {"field": "support.reading_identity"})
        if owners[0].reading_identity != support.reading_identity:
            raise _InputRefusal("provenance_mismatch",
                                {"field": "support.reading_identity"})
        if support.reading_identity_sha256 not in mask_ids:
            raise _InputRefusal("missing_provenance", {"field": "usable_mask"})
        reading = readings.get((support.reading_identity.frame,
                                support.reading_identity.reading_axis))
        if reading is None:
            raise _InputRefusal("missing_provenance", {"field": "reading"})
        evidence = reading.region_evidence
        if (support.source_region_id != evidence.source_region_id
                or support.region_mask_sha256 != evidence.mask_sha256
                or support.image_shape != (int(evidence.image_shape[0]),
                                           int(evidence.image_shape[1]))):
            raise _InputRefusal("provenance_mismatch",
                                {"field": "region_evidence"})

    if set(supports_by_digest) != {face.support_sha256
                                   for face in candidate_placement.faces}:
        raise _InputRefusal("provenance_mismatch",
                            {"field": "unreferenced_support"})


def _replay_support(support: v1.AlignedRegionSupport) -> None:
    """Rebuild every support cell and the support from its own actual fields."""
    try:
        for cell in support.cells:
            rebuilt_cell = v1.AlignedSupportCell.build(
                row=cell.row, column=cell.column,
                cell_area=Fraction(*cell.cell_area),
                region_excluded=Fraction(*cell.region_excluded),
                usable_excluded=Fraction(*cell.usable_excluded),
                frame_excluded=Fraction(*cell.frame_excluded))
            if rebuilt_cell.cell_sha256 != cell.cell_sha256:
                raise _InputRefusal("support_replay_mismatch", {"field": "cell"})
        rebuilt = v1.AlignedRegionSupport.build(
            **_record_fields(support, ("reading_binding_sha256",
                                       "support_sha256")))
    except _InputRefusal:
        raise
    except (AttributeError, TypeError, ValueError, OverflowError,
            ZeroDivisionError) as error:
        raise _InputRefusal("support_replay_mismatch",
                            {"detail": type(error).__name__}) from error
    if rebuilt.support_sha256 != support.support_sha256:
        raise _InputRefusal("support_replay_mismatch", {"field": "support"})


def _bind_supports(
    placement: plane_placement.ResolvedFacePlacement,
    supports: tuple[v1.AlignedRegionSupport, ...],
) -> tuple[tuple[Any, v1.AlignedRegionSupport], ...]:
    """Bind every face to exactly one accepted, cross-checked support."""
    anchor_digest = plane_placement.placement_anchor_digest(placement.alpha)
    by_digest: dict[str, v1.AlignedRegionSupport] = {}
    for support in supports:
        if support.support_sha256 in by_digest:
            raise _InputRefusal("provenance_mismatch",
                                {"field": "duplicate_support"})
        by_digest[support.support_sha256] = support

    bound: list[tuple[Any, v1.AlignedRegionSupport]] = []
    seen: set[str] = set()
    for face in placement.faces:
        if face.support_sha256 in seen:
            raise _InputRefusal("provenance_mismatch",
                                {"field": "duplicate_face_support"})
        seen.add(face.support_sha256)
        support = by_digest.get(face.support_sha256)
        if support is None:
            raise _InputRefusal("missing_provenance", {"field": "support"})
        if support.reading_identity_sha256 != face.reading_identity_sha256:
            raise _InputRefusal("provenance_mismatch",
                                {"field": "reading_identity"})
        if support.source_region_id != face.source_region_id:
            raise _InputRefusal("provenance_mismatch",
                                {"field": "source_region_id"})
        if support.phase_anchor_sha256 != anchor_digest:
            raise _InputRefusal("provenance_mismatch",
                                {"field": "phase_anchor_sha256"})
        try:
            distance = Fraction(support.alternative.distance_blocks)
        except (TypeError, ValueError, OverflowError) as error:
            raise _InputRefusal("support_mismatch",
                                {"field": "distance"}) from error
        if distance != Fraction(*face.d):
            raise _InputRefusal("support_mismatch", {"field": "distance"})
        if (support.column_label != face.tangent_mappings[0].local_label
                or support.row_label != face.tangent_mappings[1].local_label):
            raise _InputRefusal("support_mismatch", {"field": "labels"})
        if support.usable_mask_sha256 is None:
            raise _InputRefusal("missing_provenance", {"field": "usable_mask"})

        supported = {(cell.row, cell.column) for cell in support.cells
                     if cell.state == "supported"}
        face_cells = {(cell.row, cell.column) for cell in face.cells}
        if supported != face_cells:
            raise _InputRefusal("support_mismatch",
                                {"field": "supported_cells"})
        for cell in face.cells:
            expected_u = _cell_lower_endpoint(
                face.tangent_mappings[0].anchor,
                face.tangent_mappings[0].world_sign, int(cell.column))
            expected_v = _cell_lower_endpoint(
                face.tangent_mappings[1].anchor,
                face.tangent_mappings[1].world_sign, int(cell.row))
            if (cell.lateral_u, cell.lateral_v) != (expected_u, expected_v):
                raise _InputRefusal("replay_mismatch", {"field": "cell"})
        bound.append((face, support))

    if set(by_digest) != seen:
        raise _InputRefusal("provenance_mismatch",
                            {"field": "unreferenced_support"})
    return tuple(bound)


def _support_binding(face: Any, support: v1.AlignedRegionSupport) -> dict:
    supported = sum(1 for cell in support.cells if cell.state == "supported")
    return {
        "frame": face.frame,
        "plane_index": face.plane_index,
        "reading_axis": face.reading_axis,
        "reading_identity_sha256": face.reading_identity_sha256,
        "support_sha256": face.support_sha256,
        "face_sha256": face.face_sha256,
        "source_region_id": face.source_region_id,
        "alternative_distance": list(face.d),
        "phase_anchor_sha256": support.phase_anchor_sha256,
        "support_cells": len(support.cells),
        "supported_cells": supported,
        "unknown_cells": len(support.cells) - supported,
    }


# ---------------------------------------------------------------------------
# Source-bound replay binding and record derivation
# ---------------------------------------------------------------------------

def _replay_bound_source(
    *,
    universe: plane_placement.PlacementUniverse,
    supports: Sequence["v1.AlignedRegionSupport"],
    placement: plane_placement.ResolvedFacePlacement,
    placement_budget: plane_placement.PlacementBudget,
) -> tuple[plane_placement.ResolvedFacePlacement,
           tuple[tuple[Any, v1.AlignedRegionSupport], ...]]:
    """Validate the inputs and rerun the whole source-bound replay.

    The accepted V2 producer is rerun from the original universe, supports and
    placement budget; every candidate face and support is cross-checked against
    its source-derived counterpart; and the supports are structurally replayed.
    Nothing here consults or performs a volume budget decision.
    """
    if not isinstance(placement_budget, plane_placement.PlacementBudget):
        raise _InputRefusal("invalid_input", {"field": "placement_budget"})
    if not isinstance(universe, plane_placement.PlacementUniverse):
        raise _InputRefusal("invalid_input", {"field": "universe"})
    if not isinstance(placement, plane_placement.ResolvedFacePlacement):
        raise _InputRefusal("invalid_input", {"field": "placement"})
    if (not isinstance(supports, Sequence) or isinstance(supports, (str, bytes))
            or not supports
            or any(not isinstance(support, v1.AlignedRegionSupport)
                   for support in supports)):
        raise _InputRefusal("invalid_input", {"field": "supports"})

    ordered_supports = tuple(sorted(supports, key=v1.support_sort_key))
    support_ids = [support.support_sha256 for support in ordered_supports]
    if len(set(support_ids)) != len(support_ids):
        raise _InputRefusal("provenance_mismatch", {"field": "duplicate_support"})
    for support in ordered_supports:
        _replay_support(support)
    source_placement = _replay_source_placement(
        universe, ordered_supports, placement_budget)
    _cross_check_sources(universe, ordered_supports, source_placement, placement)
    bound = _bind_supports(source_placement, ordered_supports)
    return source_placement, bound


@dataclass(frozen=True, slots=True)
class _VolumeRecords:
    """Full geometry-only records derived from one bound source replay."""

    bindings: tuple[dict, ...]
    cells: tuple[SupportedCellObservation, ...]
    conflicts: tuple[CellConflict, ...]
    omitted: tuple[OmittedCell, ...]
    cell_bounds: tuple[tuple[int, int, int], tuple[int, int, int]] | None
    dense_bounds_cells: int
    normalization_offset: tuple[int, int, int] | None


def _derive_volume_records(
    source_placement: plane_placement.ResolvedFacePlacement,
    bound: tuple[tuple[Any, v1.AlignedRegionSupport], ...],
    counter: Any,
) -> _VolumeRecords:
    """Emit every cell, conflict and omission, ticking ``counter`` in one order.

    ``counter`` is either a stop-on-cap :class:`_Counter` (the public builder) or
    an unlimited :class:`_RecordingCounter` (the preview).  Both receive exactly
    the same ``tick``/``set_dense_bounds`` sequence, so the authoritative full
    records and the later budget decision cannot disagree about the observed work.
    """
    claims_by_coordinate: dict[tuple[int, int, int], list[CellClaim]] = {}
    omitted: list[OmittedCell] = []
    bindings: list[dict] = []
    for face, support in bound:
        counter.tick("faces")
        counter.tick("support_cells", len(support.cells))
        bindings.append(_support_binding(face, support))
        normal_axis = int(face.world_normal_axis)
        tangent_u = face.tangent_mappings[0]
        tangent_v = face.tangent_mappings[1]
        for cell in face.cells:
            coordinate = [0, 0, 0]
            coordinate[normal_axis] = int(face.normal_block)
            coordinate[int(tangent_u.world_axis)] = int(cell.lateral_u)
            coordinate[int(tangent_v.world_axis)] = int(cell.lateral_v)
            claim = CellClaim.build(
                frame=face.frame,
                plane_index=int(face.plane_index),
                reading_axis=int(face.reading_axis),
                absolute_quadrant=int(face.absolute_quadrant),
                source_region_id=face.source_region_id,
                reading_identity_sha256=face.reading_identity_sha256,
                support_sha256=face.support_sha256,
                face_sha256=face.face_sha256,
                row=int(cell.row), column=int(cell.column),
                world_normal_axis=normal_axis,
                world_normal_sign=int(face.world_normal_sign),
                outward_sign=int(face.outward_sign),
                face=int(face.face),
                normal_block=int(face.normal_block),
                tangent_u_axis=int(tangent_u.world_axis),
                lateral_u=int(cell.lateral_u),
                tangent_v_axis=int(tangent_v.world_axis),
                lateral_v=int(cell.lateral_v),
                coordinate=tuple(coordinate))
            claims_by_coordinate.setdefault(tuple(coordinate), []).append(claim)
            counter.tick("contributions")
        for cell in support.cells:
            if cell.state == "supported":
                continue
            counter.tick("omitted_unknown")
            omitted.append(OmittedCell.build(
                frame=face.frame,
                plane_index=int(face.plane_index),
                reading_axis=int(face.reading_axis),
                row=int(cell.row), column=int(cell.column),
                state=cell.state,
                evidence=cell.evidence,
                source_region_id=face.source_region_id,
                reading_identity_sha256=face.reading_identity_sha256,
                support_sha256=face.support_sha256,
                face_sha256=face.face_sha256))

    cells: list[SupportedCellObservation] = []
    conflicts: list[CellConflict] = []
    for coordinate in sorted(claims_by_coordinate):
        claims = tuple(sorted(claims_by_coordinate[coordinate],
                              key=_claim_sort_key))
        frame_ids = tuple(sorted({claim.frame for claim in claims}))
        kind = _incompatible_evidence_kind(coordinate, claims)
        cells.append(SupportedCellObservation.build(
            coordinate=coordinate, claims=claims, frame_ids=frame_ids,
            conflict_kind=kind))
        counter.tick("coordinates")
        if kind is not None:
            conflicts.append(CellConflict.build(
                coordinate=coordinate, kind=kind, claims=claims,
                frame_ids=frame_ids))
            counter.tick("conflicts")

    ordered_cells = tuple(cells)
    if ordered_cells:
        low = tuple(min(cell.coordinate[axis] for cell in ordered_cells)
                    for axis in V3_VOLUME_WORLD_AXES)
        high = tuple(max(cell.coordinate[axis] for cell in ordered_cells)
                     for axis in V3_VOLUME_WORLD_AXES)
        cell_bounds = (low, high)
        dense = 1
        for axis in V3_VOLUME_WORLD_AXES:
            dense *= high[axis] - low[axis] + 1
        normalization_offset = low
    else:
        cell_bounds = None
        dense = 0
        normalization_offset = None
    counter.set_dense_bounds(dense)
    return _VolumeRecords(
        bindings=tuple(sorted(bindings, key=_binding_sort_key)),
        cells=ordered_cells,
        conflicts=tuple(conflicts),
        omitted=tuple(sorted(omitted, key=_omitted_sort_key)),
        cell_bounds=cell_bounds,
        dense_bounds_cells=dense,
        normalization_offset=normalization_offset)


def _build_volume_result(
    source_placement: plane_placement.ResolvedFacePlacement,
    records: _VolumeRecords,
    work: VolumeWork,
) -> PlaneVolumeResult:
    return PlaneVolumeResult.build(
        schema_version=V3_VOLUME_SCHEMA,
        producer_version=V3_VOLUME_PRODUCER_VERSION,
        gauge_state=source_placement.gauge_state,
        gauge_axes=tuple(source_placement.gauge_axes),
        alpha=tuple(_pair(Fraction(*item)) for item in source_placement.alpha),
        placement_sha256=source_placement.placement_sha256,
        support_bindings=records.bindings,
        cells=records.cells,
        conflicts=records.conflicts,
        omitted=records.omitted,
        cell_bounds=records.cell_bounds,
        dense_bounds_cells=records.dense_bounds_cells,
        normalization_offset=records.normalization_offset,
        export_bounds=V3_VOLUME_EXPORT_BOUNDS,
        complete=V3_VOLUME_COMPLETE,
        work=work)


def _budget_refusal_from_stop(stop: _BudgetStop,
                              counter: _Counter) -> VolumeRefusal:
    reason = ("dense_bounds_exceeded"
              if stop.cap == "max_dense_bounds_cells" else "budget_exhausted")
    return _make_refusal(reason, {
        "cap": stop.cap, "limit": stop.limit,
        "observed": stop.observed}, counter.snapshot())


def _apply_volume_budget(budget: PlaneVolumeBudget,
                         events: Sequence[tuple]) -> VolumeRefusal | None:
    """Replay recorded work events through a stop-on-cap counter.

    The events were recorded in the exact incremental order the public builder
    uses, so a cap excess is reported at the same point with the same observed
    counter.  Returns the typed refusal, or ``None`` when the budget holds.
    """
    counter = _Counter(budget)
    try:
        for event in events:
            if event[0] == "dense":
                counter.set_dense_bounds(event[1])
            else:
                counter.tick(event[1], event[2])
    except _BudgetStop as stop:
        return _budget_refusal_from_stop(stop, counter)
    return None


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_plane_volume(
    *,
    universe: plane_placement.PlacementUniverse,
    supports: Sequence["v1.AlignedRegionSupport"],
    placement: plane_placement.ResolvedFacePlacement,
    placement_budget: plane_placement.PlacementBudget,
    budget: PlaneVolumeBudget,
    producer_version: str = V3_VOLUME_PRODUCER_VERSION,
) -> PlaneVolumeResult | VolumeRefusal:
    """Convert one source-bound placement plus its supports into V3 cells.

    The original universe, the original supports, the candidate placement and
    both budgets are required.  The accepted V2 producer is rerun from the
    original universe, supports and placement budget, and the canonical
    placement records plus every face and support digest are compared before any
    volume budget decision, so a freshly resealed placement that keeps the
    original bindings (for example a ``+100`` normal translation) is refused.
    Each supported face cell then emits exactly one adjacent block; merged
    coordinates retain all compatible claims; unknown cells are omitted with
    typed evidence; exceeding a cap or the dense bounding-box preflight discards
    everything atomically and carries no cells.
    """
    if not isinstance(budget, PlaneVolumeBudget):
        return _make_refusal("invalid_input", {"field": "budget"})
    counter = _Counter(budget)
    try:
        if type(producer_version) is not str \
                or producer_version != V3_VOLUME_PRODUCER_VERSION:
            raise _InputRefusal("invalid_input", {"field": "producer_version"})
        source_placement, bound = _replay_bound_source(
            universe=universe, supports=supports, placement=placement,
            placement_budget=placement_budget)
        records = _derive_volume_records(source_placement, bound, counter)
        work = counter.snapshot()
        return _build_volume_result(source_placement, records, work)
    except _BudgetStop as stop:
        return _budget_refusal_from_stop(stop, counter)
    except _InputRefusal as refusal:
        return _make_refusal(refusal.reason, refusal.detail)


# ---------------------------------------------------------------------------
# Explicit geometry-only BlockVolume preview adapter
# ---------------------------------------------------------------------------

def _derive_cell_geometry(
    cells: Sequence[SupportedCellObservation],
) -> tuple[tuple[tuple[int, int, int], tuple[int, int, int]] | None, int,
           tuple[int, int, int] | None]:
    """Derive cell bounds, dense size and normalization from emitted cells."""
    if not cells:
        return None, 0, None
    low = tuple(min(cell.coordinate[axis] for cell in cells)
                for axis in V3_VOLUME_WORLD_AXES)
    high = tuple(max(cell.coordinate[axis] for cell in cells)
                 for axis in V3_VOLUME_WORLD_AXES)
    dense = 1
    for axis in V3_VOLUME_WORLD_AXES:
        dense *= high[axis] - low[axis] + 1
    return (low, high), dense, low


def _preview_block_payload(value: "PreviewBlock") -> dict:
    return {
        "coordinate": list(value.coordinate),
        "frame_ids": list(value.frame_ids),
    }


@dataclass(frozen=True, slots=True)
class PreviewBlock:
    """One immutable geometry-only preview block and its real frame IDs."""

    coordinate: tuple[int, int, int]
    frame_ids: tuple[str, ...]
    block_sha256: str

    def __post_init__(self) -> None:
        if not _validate_coordinate(self.coordinate):
            raise ValueError("preview block coordinate must be three integers")
        if (not isinstance(self.frame_ids, tuple) or not self.frame_ids
                or any(type(frame) is not str or not frame
                       for frame in self.frame_ids)
                or tuple(sorted(set(self.frame_ids))) != self.frame_ids):
            raise ValueError("preview block frame IDs are malformed")
        if (not _hex64(self.block_sha256)
                or self.block_sha256 != _sha256(_preview_block_payload(self))):
            raise ValueError("preview block digest mismatch")

    @classmethod
    def build(cls, **fields: Any) -> "PreviewBlock":
        provisional = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "block_sha256", "")
        object.__setattr__(provisional, "block_sha256",
                           _sha256(_preview_block_payload(provisional)))
        cls.__post_init__(provisional)
        return provisional

    def to_record(self) -> dict:
        record = _preview_block_payload(self)
        record["block_sha256"] = self.block_sha256
        return record


def _canonical_preview_state(
    state: Any,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Validate and copy a preview state into an immutable name+properties pair.

    A list-backed ``BlockState`` is canonicalized into an immutable tuple
    snapshot rather than retained by reference, so a caller cannot mutate the
    sealed preview or its exported observations after ``preview_sha256`` is
    computed.  Any other malformed nested shape is refused.
    """
    if not isinstance(state, BlockState):
        raise _InputRefusal("preview_state_invalid", {"field": "preview_state"})
    name = state.name
    properties = state.properties
    if type(name) is not str or ":" not in name:
        raise _InputRefusal("preview_state_invalid", {"field": "name"})
    if not isinstance(properties, (list, tuple)):
        raise _InputRefusal("preview_state_invalid", {"field": "properties"})
    snapshot: list[tuple[str, str]] = []
    for item in properties:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise _InputRefusal("preview_state_invalid", {"field": "properties"})
        key, value = item
        if type(key) is not str or type(value) is not str or not key or not value:
            raise _InputRefusal("preview_state_invalid", {"field": "properties"})
        snapshot.append((key, value))
    return name, tuple(snapshot)


def _preview_state_payload(name: Any, properties: Any) -> dict:
    return {"name": name, "properties": dict(properties)}


def _preview_payload(value: "GeometryOnlyPreview") -> dict:
    return {
        "schema_version": value.schema_version,
        "producer_version": value.producer_version,
        "preview_state": _preview_state_payload(
            value.preview_state_name, value.preview_state_properties),
        "blocks": [block.to_record() for block in value.blocks],
        "normalization_offset": (None if value.normalization_offset is None
                                 else list(value.normalization_offset)),
        "export_bounds": value.export_bounds,
        "geometry_only": value.geometry_only,
        "exact_state_eligible": value.exact_state_eligible,
        "volume_sha256": value.volume_sha256,
    }


@dataclass(frozen=True, slots=True)
class GeometryOnlyPreview:
    """An immutable geometry-only preview and its exporter normalization.

    The record stores the preview state as an immutable ``name`` plus a tuple of
    property pairs, never a live ``BlockState`` reference, so a caller that
    mutates a list-backed ``BlockState`` after sealing cannot change the record or
    its digest.  It stores only an immutable tuple of :class:`PreviewBlock`
    records and never exposes a mutable ``BlockVolume``; a caller that needs an
    exporter volume calls :meth:`to_block_volume`, which builds a fresh state and
    volume, so mutating that copy cannot change this record either.
    ``exact_state_eligible`` is hard ``False`` and every produced observation is
    ``inferred`` with zero confidence, so the preview cannot be presented as M5
    material evidence or exact-state scoring.  ``normalization_offset`` is the
    world cell minimum that an exporter called with ``bounds=None`` maps to the
    local origin.
    """

    schema_version: str
    producer_version: str
    preview_state_name: str
    preview_state_properties: tuple[tuple[str, str], ...]
    blocks: tuple[PreviewBlock, ...]
    normalization_offset: tuple[int, int, int] | None
    export_bounds: None
    geometry_only: bool
    exact_state_eligible: bool
    volume_sha256: str
    preview_sha256: str

    @property
    def preview_state(self) -> BlockState:
        """A fresh immutable ``BlockState`` built from the sealed snapshot."""
        return BlockState(name=self.preview_state_name,
                          properties=self.preview_state_properties)

    def __post_init__(self) -> None:
        if (self.schema_version != V3_VOLUME_SCHEMA
                or self.producer_version != V3_VOLUME_PRODUCER_VERSION):
            raise ValueError("preview record version is unsupported")
        if (type(self.preview_state_name) is not str
                or ":" not in self.preview_state_name):
            raise ValueError("preview state name is malformed")
        if (type(self.preview_state_properties) is not tuple
                or any(type(item) is not tuple or len(item) != 2
                       or type(item[0]) is not str or type(item[1]) is not str
                       or not item[0] or not item[1]
                       for item in self.preview_state_properties)):
            raise ValueError("preview state properties are malformed")
        if _is_air_state_name(self.preview_state_name):
            raise ValueError("preview state must not be an air state")
        if (not isinstance(self.blocks, tuple)
                or any(not isinstance(block, PreviewBlock)
                       for block in self.blocks)):
            raise ValueError("preview blocks are malformed")
        coordinates = [block.coordinate for block in self.blocks]
        if (coordinates != sorted(coordinates)
                or len(set(coordinates)) != len(coordinates)):
            raise ValueError("preview blocks are not unique and canonically ordered")
        for block in self.blocks:
            if block.block_sha256 != _sha256(_preview_block_payload(block)):
                raise ValueError("preview block digest does not replay")
        if self.normalization_offset is not None \
                and not _validate_coordinate(self.normalization_offset):
            raise ValueError("preview normalization offset is malformed")
        if self.export_bounds is not V3_VOLUME_EXPORT_BOUNDS:
            raise ValueError("preview export must use bounds=None")
        if self.geometry_only is not True or self.exact_state_eligible is not False:
            raise ValueError("preview must stay geometry-only and ineligible")
        if not _hex64(self.volume_sha256):
            raise ValueError("preview volume binding is malformed")
        if (not _hex64(self.preview_sha256)
                or self.preview_sha256 != _sha256(_preview_payload(self))):
            raise ValueError("preview digest mismatch")

    @classmethod
    def build(cls, **fields: Any) -> "GeometryOnlyPreview":
        provisional = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "preview_sha256", "")
        object.__setattr__(provisional, "preview_sha256",
                           _sha256(_preview_payload(provisional)))
        cls.__post_init__(provisional)
        return provisional

    def to_record(self) -> dict:
        record = _preview_payload(self)
        record["preview_sha256"] = self.preview_sha256
        return record

    def to_block_volume(self) -> BlockVolume:
        """Return a fresh, independent mutable ``BlockVolume`` for an exporter.

        A fresh ``BlockState`` is rebuilt from the immutable snapshot on every
        call, and the preview stores no mutable volume, so mutating the returned
        state or volume cannot change this record or invalidate its digest.
        """
        state = BlockState(name=self.preview_state_name,
                           properties=self.preview_state_properties)
        volume = BlockVolume()
        for block in self.blocks:
            volume.set(block.coordinate, BlockObservation(
                state=state,
                confidence=V3_VOLUME_PREVIEW_STATE_CONFIDENCE,
                kind=ObservationKind.INFERRED,
                frame_ids=list(block.frame_ids)))
        return volume


_PREVIEW_REPLAY_ERRORS = (AttributeError, TypeError, ValueError, IndexError,
                          KeyError, OverflowError, ZeroDivisionError)


def to_geometry_only_preview(
    result: PlaneVolumeResult,
    *,
    universe: plane_placement.PlacementUniverse,
    supports: Sequence["v1.AlignedRegionSupport"],
    placement: plane_placement.ResolvedFacePlacement,
    placement_budget: plane_placement.PlacementBudget,
    budget: PlaneVolumeBudget,
    preview_state: BlockState,
    geometry_only: bool,
    producer_version: str = V3_VOLUME_PRODUCER_VERSION,
) -> GeometryOnlyPreview | VolumeRefusal:
    """Build the explicit geometry-only preview from a replayed result.

    The same source replay context is required.  The accepted build is rerun from
    the original universe, supports and placement budget, and the supplied result
    is compared with the authoritative full rebuild **before any volume budget
    decision**, so a phantom coordinate is refused as ``replay_mismatch`` even
    when the volume budget is too small for the builder to emit a result, and a
    malformed result (for example ``cells=(None,)``) becomes a typed refusal
    instead of an escaped ``AttributeError``.  Only after the result and source
    replay validate is the recorded work replayed through the caller's volume
    budget.  ``geometry_only`` must be the literal ``True``; a false or merely
    truthy value, an air preview state and any non-volume result are refused.
    The preview state is copied into an immutable name+properties snapshot, so a
    list-backed ``BlockState`` cannot be mutated into the sealed preview.  The
    preview is immutable and never exposes a mutable ``BlockVolume``.
    """
    if type(producer_version) is not str \
            or producer_version != V3_VOLUME_PRODUCER_VERSION:
        return _make_refusal("invalid_input", {"field": "producer_version"})
    if not isinstance(budget, PlaneVolumeBudget):
        return _make_refusal("invalid_input", {"field": "budget"})
    if not isinstance(result, PlaneVolumeResult):
        return _make_refusal("invalid_input", {"field": "result"})
    if geometry_only is not True:
        return _make_refusal("geometry_only_required",
                             {"field": "geometry_only"})
    try:
        preview_state_name, preview_state_properties = _canonical_preview_state(
            preview_state)
    except _InputRefusal as refusal:
        return _make_refusal(refusal.reason, refusal.detail)
    except _PREVIEW_REPLAY_ERRORS:
        return _make_refusal("preview_state_invalid",
                             {"field": "preview_state"})
    if _is_air_state_name(preview_state_name):
        return _make_refusal("air_state_forbidden", {
            "field": "preview_state", "name": preview_state_name})

    # Source replay and the authoritative full record set, with no volume budget
    # decision yet.  Every replay failure is a typed refusal before any budget.
    try:
        source_placement, bound = _replay_bound_source(
            universe=universe, supports=supports, placement=placement,
            placement_budget=placement_budget)
        recorder = _RecordingCounter()
        records = _derive_volume_records(source_placement, bound, recorder)
        authoritative = _build_volume_result(
            source_placement, records, _make_work(recorder.counts))
    except _InputRefusal as refusal:
        return _make_refusal(refusal.reason, refusal.detail)
    except _BudgetStop as stop:  # defensive: the recorder never stops
        return _make_refusal("replay_mismatch",
                             {"field": "replay", "cap": stop.cap})
    except _PREVIEW_REPLAY_ERRORS:
        return _make_refusal("replay_mismatch", {"field": "replay"})

    # Nested structure and integrity validation of the supplied result, still
    # before any volume budget decision.
    try:
        replayed = (result.volume_sha256 == authoritative.volume_sha256
                    and result.to_record() == authoritative.to_record())
    except _PREVIEW_REPLAY_ERRORS:
        return _make_refusal("replay_mismatch", {"field": "result"})
    if not replayed:
        return _make_refusal("replay_mismatch", {"field": "result"})

    cell_bounds, dense, normalization_offset = _derive_cell_geometry(
        authoritative.cells)
    if (cell_bounds != authoritative.cell_bounds
            or dense != authoritative.dense_bounds_cells
            or normalization_offset != authoritative.normalization_offset):
        return _make_refusal("replay_mismatch", {"field": "bounds"})

    # The volume budget decision, at last: replay the recorded work in the
    # builder's exact order so the refusal matches the builder byte-for-byte.
    refusal = _apply_volume_budget(budget, recorder.events)
    if refusal is not None:
        return refusal
    if dense > budget.max_dense_bounds_cells:
        return _make_refusal("dense_bounds_exceeded", {
            "cap": "max_dense_bounds_cells",
            "limit": budget.max_dense_bounds_cells,
            "observed": dense})

    try:
        blocks = tuple(PreviewBlock.build(coordinate=cell.coordinate,
                                          frame_ids=cell.frame_ids)
                       for cell in authoritative.cells)
        return GeometryOnlyPreview.build(
            schema_version=V3_VOLUME_SCHEMA,
            producer_version=V3_VOLUME_PRODUCER_VERSION,
            preview_state_name=preview_state_name,
            preview_state_properties=preview_state_properties,
            blocks=blocks,
            normalization_offset=authoritative.normalization_offset,
            export_bounds=V3_VOLUME_EXPORT_BOUNDS,
            geometry_only=True,
            exact_state_eligible=False,
            volume_sha256=authoritative.volume_sha256)
    except _PREVIEW_REPLAY_ERRORS:
        return _make_refusal("invalid_input", {"field": "preview"})
