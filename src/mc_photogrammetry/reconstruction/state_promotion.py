"""Bounded M5 state-promotion adapter.

This module is the narrow seam that turns one accepted, immutable, geometry-only
:class:`~mc_photogrammetry.reconstruction.plane_volume.PlaneVolumeResult` plus one
accepted, immutable
:class:`~mc_photogrammetry.reconstruction.appearance_state_proposals.FaceStateProposalResult`
into a typed, digest-bound set of promoted block states.  It consumes those two
records and an explicit campaign budget, and nothing else.  It performs no IO,
has no filesystem, network, process, environment or dynamic-import access, and it
never calls an exporter.

What a promotion is, and is not
-------------------------------

A :class:`PromotedBlock` binds one geometry cell's coordinate to the *exact*
state proposed for it, together with the proposal's confidence evidence, support
frame ids, provenance references, observation digests, proposal digest and
evidence digest.  Every promoted block is a *material* claim derived from a
proposed :class:`FaceStateProposal`; the adapter never infers a hidden block, air,
terrain, geometry, camera, coordinate, manifest, F3 reference, scale, bound or
manual metadata.

Input discipline
----------------

* The volume and the proposal result are re-validated at this boundary, not
  trusted: both must be immutable records, their own top-level digests must
  replay, and every nested digest (each geometry cell and every nested
  ``CellClaim`` rebuilt through ``CellClaim.build``, each conflict, each
  omission, the work record, and each per-cell proposal) is rebuilt through its
  accepted constructor before any budget is consulted.  A record built past its
  ``__post_init__`` therefore cannot ride along on a stale nested digest, and a
  forged claim field the enclosing record-level digest cannot see (an empty
  ``frame`` for example) is still a typed ``digest_mismatch``.
* Geometry cells must be unique and canonically ordered; no :class:`CellConflict`
  may be present; and the proposal cells must be *exactly* the geometry cells, so
  a missing or extraneous proposal is a typed refusal rather than a silent drop or
  an invented block.
* Every geometry cell must carry a proposal with status ``"proposed"`` and the
  enclosing result must itself have status ``"proposed"``: a result-level
  ``"review"`` is a typed ``review_abstention`` even when it carries
  individually proposed children.  Any review/abstention, an unknown geometry
  omission (unless the caller explicitly opts in), a conflict, a digest
  mismatch, an invalid/air state or a source mismatch is a typed, atomic
  review/refusal with **no partial blocks**.

Eligibility
-----------

``export_ready`` is true exactly when every geometry cell carries a proposed
state and the geometry has no conflicts; ``exact_state_eligible`` additionally
requires that there are **zero** omitted unknown cells.  Omitted unknown cells are
preserved as an explicit count and never become air or an invented block, so a
volume with omissions may be exportable material evidence while remaining
ineligible for exact-state scoring.

Preserved open metadata
-----------------------

The whole-build integer translation gauge and the normalization/scale state stay
explicitly unresolved: :data:`PROMOTION_SCALE_STATE` and
:data:`PROMOTION_NORMALIZATION_STATE` are carried as literals, the volume's
``normalization_offset`` is copied unchanged, and ``export_bounds`` is fixed at
``None`` so an exporter must derive normalization from the emitted cells alone.

Budgets
-------

``max_cells``, ``max_proposals`` and ``max_payload_bytes`` are explicit
non-negative limits.  The budget's own ``budget_sha256`` is replayed from its
declared caps before any input is inspected, so a constructor-bypassed or forged
budget is a typed ``invalid_input``/``digest_mismatch`` even when every cap is
zero.  Full input, provenance and digest validation completes before any budget
decision, so a tampered record is never reported as ``budget_exhausted``.  A cap
excess discards everything and carries no blocks.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields as _dataclass_fields
from typing import Any, Mapping

from mc_photogrammetry.domain.models import (
    BlockObservation,
    BlockState,
    BlockVolume,
    ObservationKind,
)
from mc_photogrammetry.reconstruction import appearance_state_proposals as appearance
from mc_photogrammetry.reconstruction import plane_volume as plane_volume

__all__ = [
    "PROMOTION_SCHEMA",
    "PROMOTION_PRODUCER_VERSION",
    "PROMOTION_SCALE_STATE",
    "PROMOTION_NORMALIZATION_STATE",
    "PROMOTION_EXPORT_BOUNDS",
    "PROMOTION_CONFIDENCE_POLICY",
    "PROMOTION_GAUGE_STATE",
    "PROMOTION_GAUGE_AXES",
    "PROMOTION_REVIEW_REASONS",
    "PROMOTION_REFUSAL_REASONS",
    "PROMOTION_REASONS",
    "PromotionBudget",
    "PromotionRefusal",
    "PromotedBlock",
    "PromotionResult",
    "make_promotion_budget",
    "promotion_confidence",
    "promote_plane_states",
    "promotion_result_from_record",
]

PROMOTION_SCHEMA = "m5-state-promotion-1"
PROMOTION_PRODUCER_VERSION = "m5-state-promotion-adapter-1"
PROMOTION_SCALE_STATE = "unresolved"
"""The whole-build scale/translation gauge stays unresolved and explicit."""
PROMOTION_NORMALIZATION_STATE = "unresolved"
"""Normalization is carried from the volume, never inferred here."""
PROMOTION_EXPORT_BOUNDS = None
"""An exporter must derive normalization from the emitted cells alone."""
PROMOTION_CONFIDENCE_POLICY = "margin-over-cost-separation-v1"
"""Name of the deterministic confidence derivation documented on the adapter."""
PROMOTION_GAUGE_STATE = plane_volume.V3_VOLUME_GAUGE_STATE
PROMOTION_GAUGE_AXES: tuple[int, ...] = (0, 1, 2)

PROMOTION_REVIEW_REASONS: tuple[str, ...] = (
    "review_abstention",
    "omitted_unknown",
)
"""Reasons that describe a material abstention a human/pipeline should review."""

PROMOTION_REFUSAL_REASONS: tuple[str, ...] = (
    "invalid_input",
    "source_mismatch",
    "digest_mismatch",
    "conflict",
    "missing_proposal",
    "extraneous_proposal",
    "invalid_state",
    "budget_exhausted",
)
"""Reasons that describe a structural, evidence or budget refusal."""

PROMOTION_REASONS: tuple[str, ...] = (
    PROMOTION_REVIEW_REASONS + PROMOTION_REFUSAL_REASONS
)

#: State names that would smuggle an air/terrain claim into a promotion.
_AIR_STATE_NAMES = frozenset({"air", "cave_air", "void_air"})

_BUDGET_FIELDS: tuple[str, ...] = ("max_cells", "max_proposals", "max_payload_bytes")
_BUDGET_CAPS = frozenset(_BUDGET_FIELDS)

_PROMOTION_RECORD_KEYS = frozenset({
    "schema", "producer", "volume_sha256", "proposal_result_digest",
    "gauge_state", "gauge_axes", "scale_state", "normalization_state",
    "normalization_offset", "export_bounds", "omitted_unknown", "cells",
    "export_ready", "exact_state_eligible", "max_cells", "max_proposals",
    "payload_bytes", "payload_sha256", "max_payload_bytes", "promotion_sha256",
})

_ERRORS = (
    AttributeError,
    TypeError,
    ValueError,
    IndexError,
    KeyError,
    OverflowError,
    ZeroDivisionError,
)


# ---------------------------------------------------------------------------
# Canonical JSON and small exact helpers
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


def _is_coordinate(value: Any) -> bool:
    return (isinstance(value, tuple) and len(value) == 3
            and all(type(part) is int for part in value))


def _is_real(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_air_state_name(name: Any) -> bool:
    return type(name) is str and name.rsplit(":", 1)[-1].lower() in _AIR_STATE_NAMES


def _record_fields(record: Any) -> dict:
    """All declared dataclass fields of ``record`` as a plain mapping."""
    return {field.name: getattr(record, field.name)
            for field in _dataclass_fields(record)}


class _Stop(Exception):
    """Internal typed gate signal; converted to a typed refusal by the API."""

    def __init__(self, reason: str, detail: Any):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PromotionBudget:
    """Explicit whole-request limits.  Zero is a valid, immediately-refusing cap."""

    max_cells: int
    max_proposals: int
    max_payload_bytes: int
    budget_sha256: str

    def __post_init__(self) -> None:
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 0
               for name in _BUDGET_FIELDS):
            raise ValueError("promotion budget limits must be non-negative integers")
        if (not _hex64(self.budget_sha256)
                or self.budget_sha256 != _sha256(self._payload())):
            raise ValueError("promotion budget digest mismatch")

    def _payload(self) -> dict:
        return {name: getattr(self, name) for name in _BUDGET_FIELDS}

    def to_record(self) -> dict:
        record = self._payload()
        record["budget_sha256"] = self.budget_sha256
        return record


def make_promotion_budget(**caps: int) -> PromotionBudget:
    """Build a self-validating promotion budget from explicit caps (no defaults)."""
    payload: dict[str, int] = {}
    for name in _BUDGET_FIELDS:
        if name not in caps:
            raise ValueError("promotion budget is missing a limit")
        value = caps[name]
        if type(value) is not int or value < 0:
            raise ValueError("promotion budget limits must be non-negative integers")
        payload[name] = value
    return PromotionBudget(**payload, budget_sha256=_sha256(payload))


# ---------------------------------------------------------------------------
# Typed refusal
# ---------------------------------------------------------------------------

def _refusal_payload(value: "PromotionRefusal") -> dict:
    return {
        "kind": value.kind,
        "reason": value.reason,
        "detail_sha256": value.detail_sha256,
    }


@dataclass(frozen=True, slots=True)
class PromotionRefusal:
    """One typed, atomic promotion review/refusal.  It carries no blocks."""

    kind: str
    """``"review"`` for a material abstention, ``"refusal"`` otherwise."""

    reason: str
    """One of :data:`PROMOTION_REASONS`."""

    detail: dict
    """The typed evidence for this outcome; never partial cells or blocks."""

    detail_sha256: str
    refusal_sha256: str

    def __post_init__(self) -> None:
        if self.kind not in ("review", "refusal"):
            raise ValueError("promotion refusal kind is not in the closed enum")
        if self.reason not in PROMOTION_REASONS:
            raise ValueError("promotion refusal reason is not in the closed enum")
        expected_kind = ("review" if self.reason in PROMOTION_REVIEW_REASONS
                         else "refusal")
        if self.kind != expected_kind:
            raise ValueError("promotion refusal kind disagrees with its reason")
        normalized = _json_value(self.detail)
        if not isinstance(normalized, dict):
            raise ValueError("promotion refusal detail must be a JSON object")
        encoded = _canonical_json(normalized)
        if self.detail_sha256 != hashlib.sha256(encoded.encode("utf-8")).hexdigest():
            raise ValueError("promotion refusal detail digest mismatch")
        if "cells" in normalized or "blocks" in normalized:
            raise ValueError("promotion refusal must not embed partial blocks")
        if self.reason == "budget_exhausted":
            if set(normalized) != {"cap", "limit", "observed"}:
                raise ValueError("promotion budget refusal is malformed")
            cap = normalized["cap"]
            limit = normalized["limit"]
            observed = normalized["observed"]
            if (cap not in _BUDGET_CAPS or type(limit) is not int
                    or type(observed) is not int or limit < 0
                    or observed <= limit):
                raise ValueError("promotion budget refusal evidence is malformed")
        if (not _hex64(self.refusal_sha256)
                or self.refusal_sha256 != _sha256(_refusal_payload(self))):
            raise ValueError("promotion refusal digest mismatch")
        object.__setattr__(self, "detail", normalized)

    @classmethod
    def build(cls, **fields: Any) -> "PromotionRefusal":
        provisional = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "refusal_sha256", "")
        object.__setattr__(provisional, "refusal_sha256",
                           _sha256(_refusal_payload(provisional)))
        cls.__post_init__(provisional)
        return provisional

    def to_record(self) -> dict:
        record = _refusal_payload(self)
        record["detail"] = dict(self.detail)
        record["refusal_sha256"] = self.refusal_sha256
        return record


def _refusal(reason: str, detail: Any = None, kind: str | None = None) -> PromotionRefusal:
    if reason not in PROMOTION_REASONS:
        reason = "invalid_input"
    if kind is None:
        kind = "review" if reason in PROMOTION_REVIEW_REASONS else "refusal"
    normalized = _json_value({} if detail is None else detail)
    detail_sha = hashlib.sha256(
        _canonical_json(normalized).encode("utf-8")).hexdigest()
    return PromotionRefusal.build(kind=kind, reason=reason, detail=normalized,
                                  detail_sha256=detail_sha)


# ---------------------------------------------------------------------------
# Promoted block
# ---------------------------------------------------------------------------

def _block_payload(value: "PromotedBlock") -> dict:
    return {
        "coordinate": list(value.coordinate),
        "state": {
            "name": value.state_name,
            "properties": [list(item) for item in value.state_properties],
        },
        "confidence": float(value.confidence),
        "frame_ids": list(value.frame_ids),
        "provenance": list(value.provenance),
        "observation_digests": list(value.observation_digests),
        "proposal_digest": value.proposal_digest,
        "evidence_digest": value.evidence_digest,
    }


@dataclass(frozen=True, slots=True)
class PromotedBlock:
    """One geometry cell promoted to an exact proposed state.

    The state is stored as an immutable ``name`` plus a canonical tuple of
    property pairs, never a live ``BlockState`` reference, so a caller that
    mutates a list-backed state after sealing cannot change this record.  Frame
    ids come from the proposal's support ids, and every digest is bound into
    :func:`_block_payload` so the record replays exactly.
    """

    coordinate: tuple[int, int, int]
    state_name: str
    state_properties: tuple[tuple[str, str], ...]
    confidence: float
    frame_ids: tuple[str, ...]
    provenance: tuple[str, ...]
    observation_digests: tuple[str, ...]
    proposal_digest: str
    evidence_digest: str
    block_sha256: str

    def __post_init__(self) -> None:
        if not _is_coordinate(self.coordinate):
            raise ValueError("promoted block coordinate must be three integers")
        if type(self.state_name) is not str or ":" not in self.state_name:
            raise ValueError("promoted block state name is malformed")
        if _is_air_state_name(self.state_name):
            raise ValueError("promoted block state must not be an air state")
        if (not isinstance(self.state_properties, tuple)
                or any(type(item) is not tuple or len(item) != 2
                       or type(item[0]) is not str or type(item[1]) is not str
                       or not item[0] or not item[1]
                       for item in self.state_properties)):
            raise ValueError("promoted block state properties are malformed")
        if self.state_properties != tuple(sorted(self.state_properties)):
            raise ValueError("promoted block state properties are not canonical")
        keys = [key for key, _value in self.state_properties]
        if len(set(keys)) != len(keys):
            raise ValueError("promoted block state properties repeat a key")
        if type(self.confidence) is not float or not math.isfinite(self.confidence) \
                or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("promoted block confidence must be finite and in [0, 1]")
        if (not isinstance(self.frame_ids, tuple) or not self.frame_ids
                or any(type(frame) is not str or not frame for frame in self.frame_ids)
                or tuple(sorted(set(self.frame_ids))) != self.frame_ids):
            raise ValueError("promoted block frame IDs are malformed")
        if (not isinstance(self.provenance, tuple)
                or any(type(reference) is not str or not reference
                       for reference in self.provenance)
                or tuple(sorted(set(self.provenance))) != self.provenance):
            raise ValueError("promoted block provenance is malformed")
        if (not isinstance(self.observation_digests, tuple)
                or not self.observation_digests
                or any(not _hex64(digest) for digest in self.observation_digests)
                or tuple(sorted(set(self.observation_digests)))
                != self.observation_digests):
            raise ValueError("promoted block observation digests are malformed")
        if not _hex64(self.proposal_digest) or not _hex64(self.evidence_digest):
            raise ValueError("promoted block evidence digests are malformed")
        if (not _hex64(self.block_sha256)
                or self.block_sha256 != _sha256(_block_payload(self))):
            raise ValueError("promoted block digest mismatch")

    @classmethod
    def build(cls, **fields: Any) -> "PromotedBlock":
        provisional = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "block_sha256", "")
        object.__setattr__(provisional, "block_sha256",
                           _sha256(_block_payload(provisional)))
        cls.__post_init__(provisional)
        return provisional

    def state(self) -> BlockState:
        """A fresh immutable ``BlockState`` built from the sealed snapshot."""
        return BlockState(name=self.state_name, properties=self.state_properties)

    def to_record(self) -> dict:
        record = _block_payload(self)
        record["block_sha256"] = self.block_sha256
        return record


# ---------------------------------------------------------------------------
# Promotion result
# ---------------------------------------------------------------------------

def _result_payload_values(
    *,
    schema_version: str,
    producer_version: str,
    volume_sha256: str,
    proposal_result_digest: str,
    gauge_state: str,
    gauge_axes: tuple[int, ...],
    scale_state: str,
    normalization_state: str,
    normalization_offset: tuple[int, int, int] | None,
    export_bounds: None,
    omitted_unknown: int,
    cells: tuple[PromotedBlock, ...],
    export_ready: bool,
    exact_state_eligible: bool,
    max_cells: int,
    max_proposals: int,
) -> dict:
    return {
        "schema": schema_version,
        "producer": producer_version,
        "volume_sha256": volume_sha256,
        "proposal_result_digest": proposal_result_digest,
        "gauge_state": gauge_state,
        "gauge_axes": list(gauge_axes),
        "scale_state": scale_state,
        "normalization_state": normalization_state,
        "normalization_offset": (None if normalization_offset is None
                                 else list(normalization_offset)),
        "export_bounds": export_bounds,
        "omitted_unknown": int(omitted_unknown),
        "cells": [cell.to_record() for cell in cells],
        "export_ready": bool(export_ready),
        "exact_state_eligible": bool(exact_state_eligible),
        "max_cells": int(max_cells),
        "max_proposals": int(max_proposals),
    }


def _result_payload(result: "PromotionResult") -> dict:
    return _result_payload_values(
        schema_version=result.schema_version,
        producer_version=result.producer_version,
        volume_sha256=result.volume_sha256,
        proposal_result_digest=result.proposal_result_digest,
        gauge_state=result.gauge_state,
        gauge_axes=result.gauge_axes,
        scale_state=result.scale_state,
        normalization_state=result.normalization_state,
        normalization_offset=result.normalization_offset,
        export_bounds=result.export_bounds,
        omitted_unknown=result.omitted_unknown,
        cells=result.cells,
        export_ready=result.export_ready,
        exact_state_eligible=result.exact_state_eligible,
        max_cells=result.max_cells,
        max_proposals=result.max_proposals,
    )


@dataclass(frozen=True, slots=True)
class PromotionResult:
    """The immutable, digest-bound outcome of one bounded promotion call.

    The record binds the exact geometry source (``volume_sha256``), the exact
    proposal source (``proposal_result_digest``), every per-cell proposal digest,
    state payload, frame/provenance evidence, observation digest, the carried
    ``normalization_offset`` and ``export_bounds=None``.  It never exposes a
    mutable ``BlockVolume``; a caller that needs one calls
    :meth:`to_block_volume`, which builds a fresh volume and fresh states on every
    call.
    """

    schema_version: str
    producer_version: str
    volume_sha256: str
    proposal_result_digest: str
    gauge_state: str
    gauge_axes: tuple[int, ...]
    scale_state: str
    normalization_state: str
    normalization_offset: tuple[int, int, int] | None
    export_bounds: None
    omitted_unknown: int
    cells: tuple[PromotedBlock, ...]
    export_ready: bool
    exact_state_eligible: bool
    max_cells: int
    max_proposals: int
    max_payload_bytes: int
    payload_bytes: int
    payload_sha256: str
    promotion_sha256: str

    def __post_init__(self) -> None:
        if (self.schema_version != PROMOTION_SCHEMA
                or self.producer_version != PROMOTION_PRODUCER_VERSION):
            raise ValueError("promotion record version is unsupported")
        if not _hex64(self.volume_sha256) or not _hex64(self.proposal_result_digest):
            raise ValueError("promotion source digests are malformed")
        if self.gauge_state != PROMOTION_GAUGE_STATE \
                or tuple(self.gauge_axes) != PROMOTION_GAUGE_AXES:
            raise ValueError("promotion gauge state is unsupported")
        if (self.scale_state != PROMOTION_SCALE_STATE
                or self.normalization_state != PROMOTION_NORMALIZATION_STATE):
            raise ValueError("promotion scale/normalization state must stay unresolved")
        if self.normalization_offset is not None \
                and not _is_coordinate(self.normalization_offset):
            raise ValueError("promotion normalization offset is malformed")
        if self.export_bounds is not PROMOTION_EXPORT_BOUNDS:
            raise ValueError("promotion export must use bounds=None")
        if type(self.omitted_unknown) is not int or self.omitted_unknown < 0:
            raise ValueError("promotion omitted count must be a non-negative integer")
        if (not isinstance(self.cells, tuple)
                or any(not isinstance(cell, PromotedBlock) for cell in self.cells)):
            raise ValueError("promotion cells are malformed")
        coordinates = [cell.coordinate for cell in self.cells]
        if (coordinates != sorted(coordinates)
                or len(set(coordinates)) != len(coordinates)):
            raise ValueError("promotion cells are not unique and canonically ordered")
        for cell in self.cells:
            if cell.block_sha256 != _sha256(_block_payload(cell)):
                raise ValueError("promotion cell digest does not replay")
        if self.export_ready is not True:
            raise ValueError("an emitted promotion is always export-ready")
        if type(self.exact_state_eligible) is not bool:
            raise ValueError("promotion exact-state flag must be boolean")
        if self.exact_state_eligible != (self.export_ready and self.omitted_unknown == 0):
            raise ValueError("promotion exact-state eligibility disagrees with its omissions")
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 0
               for name in _BUDGET_FIELDS):
            raise ValueError("promotion result budgets are malformed")
        if type(self.payload_bytes) is not int or self.payload_bytes < 0:
            raise ValueError("promotion payload byte count is malformed")
        payload = _result_payload(self)
        encoded = _canonical_json(payload)
        if self.payload_bytes != len(encoded.encode("utf-8")):
            raise ValueError("promotion payload byte count mismatch")
        if self.payload_sha256 != hashlib.sha256(encoded.encode("utf-8")).hexdigest():
            raise ValueError("promotion payload digest mismatch")
        if (not _hex64(self.promotion_sha256)
                or self.promotion_sha256 != _sha256({
                    **payload,
                    "payload_sha256": self.payload_sha256,
                    "max_payload_bytes": self.max_payload_bytes,
                })):
            raise ValueError("promotion digest mismatch")

    @classmethod
    def build(cls, **fields: Any) -> "PromotionResult":
        provisional = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        payload = _result_payload(provisional)
        encoded = _canonical_json(payload)
        object.__setattr__(provisional, "payload_bytes", len(encoded.encode("utf-8")))
        object.__setattr__(provisional, "payload_sha256",
                           hashlib.sha256(encoded.encode("utf-8")).hexdigest())
        object.__setattr__(provisional, "promotion_sha256", _sha256({
            **payload,
            "payload_sha256": provisional.payload_sha256,
            "max_payload_bytes": provisional.max_payload_bytes,
        }))
        cls.__post_init__(provisional)
        return provisional

    def to_record(self) -> dict:
        record = _result_payload(self)
        record["payload_bytes"] = self.payload_bytes
        record["payload_sha256"] = self.payload_sha256
        record["max_payload_bytes"] = self.max_payload_bytes
        record["promotion_sha256"] = self.promotion_sha256
        return record

    def to_block_volume(self) -> BlockVolume:
        """Return a fresh, independent mutable ``BlockVolume`` for an exporter.

        Every call rebuilds a fresh ``BlockState`` and a fresh ``frame_ids`` list
        and stores no mutable volume on this record, so mutating the returned
        volume or state cannot change this record or invalidate its digest.  Each
        observation is ``observed`` with the deterministic proposal-evidence
        confidence.  The adapter never calls an exporter itself.
        """
        volume = BlockVolume()
        for block in self.cells:
            volume.set(block.coordinate, BlockObservation(
                state=block.state(),
                confidence=block.confidence,
                kind=ObservationKind.OBSERVED,
                frame_ids=list(block.frame_ids)))
        return volume


# ---------------------------------------------------------------------------
# Strict record reconstruction
# ---------------------------------------------------------------------------

def _block_from_record(entry: Any) -> "PromotedBlock":
    """Rebuild one serialized promoted block through its accepted constructor."""
    if not isinstance(entry, Mapping):
        raise ValueError("promotion record cell must be a mapping")
    state = entry["state"]
    if not isinstance(state, Mapping):
        raise ValueError("promotion record cell state must be a mapping")
    properties = state["properties"]
    if not isinstance(properties, (list, tuple)):
        raise ValueError("promotion record cell properties must be a sequence")
    rebuilt: list[tuple[Any, Any]] = []
    for item in properties:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("promotion record cell property must be a pair")
        rebuilt.append((item[0], item[1]))
    return PromotedBlock.build(
        coordinate=tuple(entry["coordinate"]),
        state_name=state["name"],
        state_properties=tuple(rebuilt),
        confidence=entry["confidence"],
        frame_ids=tuple(entry["frame_ids"]),
        provenance=tuple(entry["provenance"]),
        observation_digests=tuple(entry["observation_digests"]),
        proposal_digest=entry["proposal_digest"],
        evidence_digest=entry["evidence_digest"],
        block_sha256=entry["block_sha256"],
    )


def promotion_result_from_record(record: Mapping[str, Any]) -> "PromotionResult":
    """Rebuild one serialized promotion record and replay every nested digest.

    The record is the canonical mapping produced by
    :meth:`PromotionResult.to_record`.  Each nested cell is rebuilt through
    :meth:`PromotedBlock.build` (which re-runs the constructor and recomputes
    ``block_sha256``), the enclosing result is rebuilt through
    :meth:`PromotionResult.build` (which recomputes ``payload_bytes``,
    ``payload_sha256`` and ``promotion_sha256``), and the rebuilt record must
    then serialize back to exactly the provided record.  A stale or resealed
    nested digest, a forged field the enclosing digest cannot see, a missing
    field or an unknown field is therefore a ``ValueError`` rather than a
    partially trusted record.
    """
    if not isinstance(record, Mapping):
        raise ValueError("promotion record must be a mapping")
    if set(record) != _PROMOTION_RECORD_KEYS:
        raise ValueError("promotion record key set is not canonical")
    fields = dict(record)
    try:
        fields["schema_version"] = fields.pop("schema")
        fields["producer_version"] = fields.pop("producer")
        cells = fields.get("cells")
        if not isinstance(cells, (list, tuple)):
            raise ValueError("promotion record cells must be a sequence")
        fields["cells"] = tuple(_block_from_record(entry) for entry in cells)
        offset = fields.get("normalization_offset")
        if offset is not None:
            fields["normalization_offset"] = tuple(offset)
        axes = fields.get("gauge_axes")
        if axes is not None:
            fields["gauge_axes"] = tuple(axes)
        result = PromotionResult.build(**fields)
    except _ERRORS as error:
        raise ValueError("promotion record does not rebuild") from error
    if _canonical_json(result.to_record()) != _canonical_json(dict(record)):
        raise ValueError("promotion record does not replay exactly")
    return result


# ---------------------------------------------------------------------------
# Confidence derivation
# ---------------------------------------------------------------------------

def promotion_confidence(best_cost: Any, margin: Any) -> float:
    """Deterministic, bounded confidence derived from one proposal's evidence.

    ``separation`` is the proposal's own margin over the runner-up relative to
    ``margin + best_cost`` (``1.0`` for the infinite single-candidate margin and
    ``0.0`` for a zero margin), and ``quality`` is the monotone ``1 / (1 +
    best_cost)``.  The product is clamped to ``[0, 1]`` and is finite for every
    accepted proposed proposal.  The derivation is named by
    :data:`PROMOTION_CONFIDENCE_POLICY` and never consults anything but the
    proposal's own aggregate costs.
    """
    if not _is_real(best_cost) or not math.isfinite(float(best_cost)) \
            or float(best_cost) < 0.0:
        raise _Stop("invalid_state", {"field": "best_cost"})
    if not _is_real(margin) or math.isnan(float(margin)) or float(margin) < 0.0:
        raise _Stop("invalid_state", {"field": "margin"})
    best = float(best_cost)
    separation = 1.0 if math.isinf(float(margin)) else (
        0.0 if float(margin) == 0.0 else float(margin) / (float(margin) + best))
    value = separation * (1.0 / (1.0 + best))
    if not math.isfinite(value):
        raise _Stop("invalid_state", {"field": "confidence"})
    return float(min(1.0, max(0.0, value)))


# ---------------------------------------------------------------------------
# Input validation and replay
# ---------------------------------------------------------------------------

def _state_snapshot(state: Any) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Validate and copy one proposed state into an immutable canonical snapshot."""
    if not isinstance(state, BlockState):
        raise _Stop("invalid_state", {"field": "state"})
    name = state.name
    if type(name) is not str or ":" not in name:
        raise _Stop("invalid_state", {"field": "state_name"})
    if _is_air_state_name(name):
        raise _Stop("invalid_state", {"field": "state_air"})
    properties = state.properties
    if not isinstance(properties, (tuple, list)):
        raise _Stop("invalid_state", {"field": "state_properties"})
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in properties:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise _Stop("invalid_state", {"field": "state_properties"})
        key, value = item
        if type(key) is not str or type(value) is not str or not key or not value:
            raise _Stop("invalid_state", {"field": "state_properties"})
        if key in seen:
            raise _Stop("invalid_state", {"field": "state_property_key"})
        seen.add(key)
        pairs.append((key, value))
    return name, tuple(sorted(pairs))


def _check_source(volume: Any, proposals: Any) -> None:
    """Version, immutability and source-shape checks.  No digest work, no budget."""
    if type(volume.cells) is not tuple or type(volume.conflicts) is not tuple \
            or type(volume.omitted) is not tuple \
            or type(volume.support_bindings) is not tuple:
        raise _Stop("invalid_input", {"field": "volume_tuples"})
    if type(proposals.proposals) is not tuple:
        raise _Stop("invalid_input", {"field": "proposal_tuple"})
    if (volume.schema_version != plane_volume.V3_VOLUME_SCHEMA
            or volume.producer_version != plane_volume.V3_VOLUME_PRODUCER_VERSION):
        raise _Stop("source_mismatch", {"field": "volume_version"})
    if volume.gauge_state != PROMOTION_GAUGE_STATE \
            or tuple(volume.gauge_axes) != PROMOTION_GAUGE_AXES:
        raise _Stop("source_mismatch", {"field": "volume_gauge"})
    if volume.export_bounds is not None or volume.complete is not False:
        raise _Stop("source_mismatch", {"field": "volume_export"})
    if proposals.status == appearance.REFUSED:
        raise _Stop("source_mismatch", {
            "field": "proposal_status", "budget": proposals.budget})
    if proposals.status not in (appearance.PROPOSED, appearance.REVIEW):
        raise _Stop("source_mismatch", {"field": "proposal_status"})


def _check_geometry(volume: Any) -> tuple[tuple[int, int, int], ...]:
    """Require conflict-free, unique, canonically ordered geometry cells."""
    if volume.conflicts:
        raise _Stop("conflict", {"geometry_conflicts": len(volume.conflicts)})
    coordinates = tuple(cell.coordinate for cell in volume.cells)
    if coordinates != tuple(sorted(coordinates)) \
            or len(set(coordinates)) != len(coordinates):
        raise _Stop("digest_mismatch", {"field": "geometry_cells"})
    flagged = sum(1 for cell in volume.cells if cell.conflict_kind is not None)
    if flagged:
        raise _Stop("conflict", {"flagged_cells": flagged})
    return coordinates


def _check_proposals(
    proposals: Any,
    coordinates: tuple[tuple[int, int, int], ...],
) -> dict[tuple[int, int, int], tuple[Any, str, tuple[tuple[str, str], ...], float]]:
    """Require the proposal cell set to equal the geometry cell set exactly."""
    if not proposals.proposals:
        raise _Stop("review_abstention", {
            "field": "result", "reason": proposals.reason})
    # An honest producer sets ``PROPOSED`` iff at least one child is proposed, so
    # a result-level review carrying any proposed child is a hostile record that
    # must be refused before its cell set is even compared.  An honest review
    # (only review children) falls through to its exact per-cell evidence below.
    if proposals.status != appearance.PROPOSED and any(
            proposal.status == appearance.PROPOSED
            for proposal in proposals.proposals):
        raise _Stop("review_abstention", {
            "field": "result_status", "status": proposals.status,
            "reason": proposals.reason})
    cells = [proposal.cell for proposal in proposals.proposals]
    if len(set(cells)) != len(cells):
        raise _Stop("extraneous_proposal", {"field": "duplicate_cell"})
    geometry = set(coordinates)
    proposed = set(cells)
    missing = geometry - proposed
    if missing:
        raise _Stop("missing_proposal", {"missing": len(missing)})
    extraneous = proposed - geometry
    if extraneous:
        raise _Stop("extraneous_proposal", {"extraneous": len(extraneous)})

    selected: dict[
        tuple[int, int, int],
        tuple[Any, str, tuple[tuple[str, str], ...], float],
    ] = {}
    for proposal in proposals.proposals:
        if proposal.status != appearance.PROPOSED:
            raise _Stop("review_abstention", {
                "cell": list(proposal.cell), "reason": proposal.reason})
        name, properties = _state_snapshot(proposal.state)
        confidence = promotion_confidence(proposal.best_cost, proposal.margin)
        selected[proposal.cell] = (proposal, name, properties, confidence)
    return selected


def _replay_claims(claims: Any) -> tuple:
    """Rebuild every nested claim through its accepted constructor.

    A claim built past ``CellClaim.__post_init__`` can carry a forged field (an
    empty ``frame``, for example) while its stored digest still replays, so a
    record-only rebuild of the enclosing cell would accept it.  Rebuilding each
    claim through :meth:`plane_volume.CellClaim.build` re-runs the constructor
    validation that no nested digest check can replace.
    """
    return tuple(
        plane_volume.CellClaim.build(**_record_fields(claim))
        for claim in claims)


def _replay_volume(volume: Any) -> None:
    """Rebuild every nested volume record through its accepted constructor."""
    try:
        cells = tuple(
            plane_volume.SupportedCellObservation.build(**{
                **_record_fields(cell),
                "claims": _replay_claims(cell.claims),
            })
            for cell in volume.cells)
        conflicts = tuple(
            plane_volume.CellConflict.build(**{
                **_record_fields(conflict),
                "claims": _replay_claims(conflict.claims),
            })
            for conflict in volume.conflicts)
        omitted = tuple(
            plane_volume.OmittedCell.build(**_record_fields(record))
            for record in volume.omitted)
        work = plane_volume.VolumeWork(**_record_fields(volume.work))
        rebuilt = plane_volume.PlaneVolumeResult.build(**{
            **_record_fields(volume),
            "cells": cells,
            "conflicts": conflicts,
            "omitted": omitted,
            "work": work,
        })
    except _ERRORS as error:
        raise _Stop("digest_mismatch", {
            "field": "volume", "detail": type(error).__name__}) from error
    if rebuilt.volume_sha256 != volume.volume_sha256 \
            or rebuilt.to_record() != volume.to_record():
        raise _Stop("digest_mismatch", {"field": "volume"})


def _replay_proposals(result: Any) -> None:
    """Rebuild every nested proposal through its accepted constructor."""
    try:
        proposals = tuple(
            appearance.FaceStateProposal(**_record_fields(proposal))
            for proposal in result.proposals)
        rebuilt = appearance.FaceStateProposalResult(**{
            **_record_fields(result),
            "proposals": proposals,
        })
    except _ERRORS as error:
        raise _Stop("digest_mismatch", {
            "field": "proposals", "detail": type(error).__name__}) from error
    if rebuilt.result_digest != result.result_digest \
            or rebuilt.proposals != result.proposals:
        raise _Stop("digest_mismatch", {"field": "proposals"})


def _replay_budget(budget: "PromotionBudget") -> None:
    """Replay a promotion budget's own digest before any input or budget work.

    A budget built past ``PromotionBudget.__post_init__`` can carry forged caps
    or a stale ``budget_sha256``.  Rebuilding the digest from the declared caps
    makes such a record a typed ``invalid_input``/``digest_mismatch`` even when
    every cap is zero, so a hostile budget can never reach a budget decision.
    """
    try:
        values = {name: getattr(budget, name) for name in _BUDGET_FIELDS}
        stored = budget.budget_sha256
    except _ERRORS as error:
        raise _Stop("invalid_input", {
            "field": "budget", "detail": type(error).__name__}) from error
    if any(type(values[name]) is not int or values[name] < 0
           for name in _BUDGET_FIELDS):
        raise _Stop("invalid_input", {"field": "budget"})
    if stored != _sha256(values):
        raise _Stop("digest_mismatch", {"field": "budget"})


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------

def _promoted_block(
    proposal: Any,
    state_name: str,
    state_properties: tuple[tuple[str, str], ...],
    confidence: float,
) -> PromotedBlock:
    return PromotedBlock.build(
        coordinate=proposal.cell,
        state_name=state_name,
        state_properties=state_properties,
        confidence=confidence,
        frame_ids=tuple(proposal.support_frame_ids),
        provenance=tuple(proposal.provenance),
        observation_digests=tuple(proposal.observation_digests),
        proposal_digest=proposal.proposal_digest,
        evidence_digest=proposal.evidence_digest,
    )


def _run_promotion(
    volume: Any,
    proposals: Any,
    budget: PromotionBudget,
    allow_omitted_unknown: bool,
) -> PromotionResult | PromotionRefusal:
    _check_source(volume, proposals)
    coordinates = _check_geometry(volume)
    selected = _check_proposals(proposals, coordinates)

    omitted_unknown = len(volume.omitted)
    if omitted_unknown and not allow_omitted_unknown:
        raise _Stop("omitted_unknown", {"omitted_unknown": omitted_unknown})

    # All nested digests replay before any budget decision.
    _replay_volume(volume)
    _replay_proposals(proposals)

    if len(coordinates) > budget.max_cells:
        raise _Stop("budget_exhausted", {
            "cap": "max_cells", "limit": budget.max_cells,
            "observed": len(coordinates)})
    if len(proposals.proposals) > budget.max_proposals:
        raise _Stop("budget_exhausted", {
            "cap": "max_proposals", "limit": budget.max_proposals,
            "observed": len(proposals.proposals)})

    blocks = tuple(
        _promoted_block(selected[cell.coordinate][0],
                        selected[cell.coordinate][1],
                        selected[cell.coordinate][2],
                        selected[cell.coordinate][3])
        for cell in volume.cells)
    payload = _result_payload_values(
        schema_version=PROMOTION_SCHEMA,
        producer_version=PROMOTION_PRODUCER_VERSION,
        volume_sha256=volume.volume_sha256,
        proposal_result_digest=proposals.result_digest,
        gauge_state=volume.gauge_state,
        gauge_axes=tuple(volume.gauge_axes),
        scale_state=PROMOTION_SCALE_STATE,
        normalization_state=PROMOTION_NORMALIZATION_STATE,
        normalization_offset=volume.normalization_offset,
        export_bounds=PROMOTION_EXPORT_BOUNDS,
        omitted_unknown=omitted_unknown,
        cells=blocks,
        export_ready=True,
        exact_state_eligible=(omitted_unknown == 0),
        max_cells=budget.max_cells,
        max_proposals=budget.max_proposals,
    )
    payload_bytes = len(_canonical_json(payload).encode("utf-8"))
    if payload_bytes > budget.max_payload_bytes:
        raise _Stop("budget_exhausted", {
            "cap": "max_payload_bytes", "limit": budget.max_payload_bytes,
            "observed": payload_bytes})

    return PromotionResult.build(
        schema_version=PROMOTION_SCHEMA,
        producer_version=PROMOTION_PRODUCER_VERSION,
        volume_sha256=volume.volume_sha256,
        proposal_result_digest=proposals.result_digest,
        gauge_state=volume.gauge_state,
        gauge_axes=tuple(volume.gauge_axes),
        scale_state=PROMOTION_SCALE_STATE,
        normalization_state=PROMOTION_NORMALIZATION_STATE,
        normalization_offset=volume.normalization_offset,
        export_bounds=PROMOTION_EXPORT_BOUNDS,
        omitted_unknown=omitted_unknown,
        cells=blocks,
        export_ready=True,
        exact_state_eligible=(omitted_unknown == 0),
        max_cells=budget.max_cells,
        max_proposals=budget.max_proposals,
        max_payload_bytes=budget.max_payload_bytes)


def promote_plane_states(
    volume: plane_volume.PlaneVolumeResult,
    proposals: appearance.FaceStateProposalResult,
    *,
    budget: PromotionBudget,
    allow_omitted_unknown: bool = False,
) -> PromotionResult | PromotionRefusal:
    """Promote one accepted geometry volume with one accepted proposal result.

    ``volume`` must be an accepted, replay-valid ``PlaneVolumeResult`` and
    ``proposals`` an accepted, replay-valid ``FaceStateProposalResult`` whose
    proposal cells are exactly the geometry cells.  The budget's own digest is
    replayed first, so a constructor-bypassed or forged budget is a typed
    ``invalid_input``/``digest_mismatch`` rather than a budget decision.  Every
    validation, provenance and nested-digest check completes before any budget
    is consulted; a malformed or tampered record is therefore a typed refusal
    rather than ``budget_exhausted``.

    A result-level or per-cell review/abstention, a conflict, a missing or
    extraneous proposal, an invalid or air state, a digest mismatch, a source
    mismatch, an unknown geometry omission (unless ``allow_omitted_unknown`` is
    the literal ``True``) and any exhausted budget all return a typed, atomic
    :class:`PromotionRefusal` that carries no blocks.  A result-level ``review``
    is refused even when it carries individually proposed children.  On success
    the returned :class:`PromotionResult` is immutable and digest-bound;
    ``to_block_volume`` produces a fresh mutable exporter volume on demand.
    """
    if not isinstance(budget, PromotionBudget):
        return _refusal("invalid_input", {"field": "budget"})
    try:
        _replay_budget(budget)
    except _Stop as stop:
        return _refusal(stop.reason, stop.detail)
    if type(allow_omitted_unknown) is not bool:
        return _refusal("invalid_input", {"field": "allow_omitted_unknown"})
    if not isinstance(volume, plane_volume.PlaneVolumeResult):
        return _refusal("invalid_input", {"field": "volume"})
    if not isinstance(proposals, appearance.FaceStateProposalResult):
        return _refusal("invalid_input", {"field": "proposals"})
    try:
        return _run_promotion(volume, proposals, budget, allow_omitted_unknown)
    except _Stop as stop:
        return _refusal(stop.reason, stop.detail)
    except _ERRORS as error:
        return _refusal("invalid_input", {
            "field": "input", "detail": type(error).__name__})
