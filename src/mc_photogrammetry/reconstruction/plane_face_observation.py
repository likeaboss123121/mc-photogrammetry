"""M5 plane-homography face-observation bridge.

This module implements one isolated seam only: turn an **accepted** M4 V3
:class:`~plane_volume.PlaneVolumeResult` plus the original source records it was
replayed from into deterministic, immutable
:class:`~face_observation.FaceObservation` records.

The contract is the plane-homography route agreed after the first bridge
attempt stopped on ambiguity (``docs/reports/M5-plane-face-observation-bridge-
2026-09-19.md`` and the ``AGENT_COMMS.md`` entries of 2026-09-19):

* **Source-bound inputs.**  The bridge requires the accepted V3 result *and* the
  original :class:`~plane_placement.PlacementUniverse`, the original V1
  :class:`~plane_region_support.AlignedRegionSupport` records, the solved
  :class:`~plane_placement.ResolvedFacePlacement`, the placement budget and the
  volume budget the accepted result was built with.  It reruns the accepted V3
  builder from those originals and refuses any result whose canonical record does
  not replay byte for byte, before any bridge budget decision.
* **Per-frame context.**  For every frame the bridge additionally requires an
  immutable :class:`FrameRectificationContext`: the frame's retained image, its
  usable mask, the upstream image digest the original supports carry, and one
  typed :class:`ReadingRectification` per reading axis.  The context is
  digest-bound to the original records: the rectification must equal the original
  reading's own :class:`~lattice_planes.PlaneRectification`, the support
  calibration must replay ``_calibration_from_reading``, the usable-mask digest
  must equal the support's own ``usable_mask_sha256``, the decoded source-region
  mask must equal the support's ``region_mask_sha256``, and the declared upstream
  image digest must equal the support's ``image_sha256``.  The retained pixels
  carry their own recomputed digest so the exact bytes consumed are identified.
* **Plane homography.**  Each claim's image quad is rebuilt from the bound
  support's ``origin_uv``, the reading calibration, the claim's ``row``/``column``
  and the retained distance by V1's own exact-rational homography/projection
  (:func:`plane_region_support._exact_homography` /
  :func:`plane_region_support._project`).  The quad is rectified with the accepted
  :func:`face_sampling.rectify_face_valid` helper, which is also what defines the
  coverage.
* **Coverage.**  Coverage is the rectified intersection of the decoded
  source-region mask, the frame's usable mask and the in-frame pixels.  The
  source-space mask is ``region & usable`` over the image, and the accepted
  rectifier fills everything outside the frame with zero, so out-of-frame pixels
  contribute no coverage.  ``FaceObservation.usable_coverage`` is the mean of the
  retained fractional coverage; ``valid``/``coverage`` are the rectifier's own
  per-pixel masks.  No cube occupancy, no ``sample_visible_faces`` and no
  occlusion model is consulted anywhere in this bridge.
* **Slots.**  Six world-face slots are derived from the claim's outward geometry:
  the outward world normal is ``outward_sign * e_{world_normal_axis}`` and maps to
  exactly one canonical :data:`~face_observation.NORMAL_OF_SLOT` slot.  The table
  is total and injective, and the accepted V3 replay guarantees every claim's
  normal axis/sign before the bridge derives anything, so the derived slot can
  never disagree with its outward geometry inside this bridge.  A hand-built
  :class:`FaceObservationBinding` whose slot contradicts its own normal is
  rejected by the record constructor itself (``ValueError``), not by a
  declared-but-unreachable bridge refusal.
* **Records.**  One immutable :class:`FaceObservation` is emitted per
  (cell, physical slot, frame claim), with ``support_frames`` restricted to the
  claims of that same cell *and* slot.  Unknown (V3 omitted) cells, conflicting
  cells and cell/slot pairs without a claim are omitted as typed
  :class:`FaceObservationReview` records instead.
* **Budgets.**  Explicit positive caps exist for claims, frames, rectified
  pixels, payload bytes and replay checks.  Every input, provenance, replay and
  context binding is completed before any budget decision, and exceeding a cap
  discards every partial record atomically with an observed-counter-only refusal.

The module defines no block material, no state fusion, no ``BlockVolume``/
``BlockState``, no export, no CLI/UI, no manual metadata and no persistence, and
it imports no export, exporter, CLI, UI or project-store module.  It performs no
IO and has no filesystem, network, process, environment or dynamic-import access.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence

import numpy as np

from mc_photogrammetry.blocks.model_atlas import FACE_BASIS, FACE_SLOTS
from mc_photogrammetry.reconstruction import face_sampling
from mc_photogrammetry.reconstruction import lattice_planes
from mc_photogrammetry.reconstruction import plane_placement
from mc_photogrammetry.reconstruction import plane_region_support as v1
from mc_photogrammetry.reconstruction import plane_volume as pv
from mc_photogrammetry.reconstruction.face_observation import (
    NORMAL_OF_SLOT,
    FaceObservation,
)


__all__ = [
    "M5_FACE_OBSERVATION_SCHEMA",
    "M5_FACE_OBSERVATION_PRODUCER_VERSION",
    "M5_FACE_OBSERVATION_COVERAGE_POLICY",
    "M5_FACE_OBSERVATION_REFUSAL_REASONS",
    "M5_FACE_OBSERVATION_REVIEW_REASONS",
    "M5_FACE_OBSERVATION_REVIEW_SCOPES",
    "SLOT_OF_NORMAL",
    "PlaneFaceObservationBudget",
    "PlaneFaceObservationWork",
    "PlaneFaceObservationRefusal",
    "ReadingRectification",
    "FrameRectificationContext",
    "FaceObservationBinding",
    "FaceObservationReview",
    "PlaneFaceObservationResult",
    "make_face_observation_budget",
    "slot_of_normal",
    "frame_context_from_readings",
    "face_observation_sha256",
    "face_observation_sort_key",
    "canonical_face_observation_order",
    "build_plane_face_observations",
]

M5_FACE_OBSERVATION_SCHEMA = "m5-plane-face-observation-1"
M5_FACE_OBSERVATION_PRODUCER_VERSION = "m5-plane-face-observation-bridge-1"
M5_FACE_OBSERVATION_COVERAGE_POLICY = (
    "rectified_source_region_usable_mask_in_frame_intersection")
#: Closed vocabulary of request-level, atomic refusals.
M5_FACE_OBSERVATION_REFUSAL_REASONS: tuple[str, ...] = (
    "invalid_input",
    "non_volume_input",
    "missing_provenance",
    "provenance_mismatch",
    "replay_mismatch",
    "volume_replay_refused",
    "missing_frame_context",
    "missing_rectification",
    "frame_context_mismatch",
    "unreferenced_frame",
    "budget_exhausted",
)
#: Closed vocabulary of per-cell/per-slot/per-claim omission records.
M5_FACE_OBSERVATION_REVIEW_REASONS: tuple[str, ...] = (
    "unknown_cell",
    "conflict_cell",
    "unsupported_slot",
    "degenerate_projection",
    "no_source_pixels",
    "off_frame",
    "empty_coverage",
)
M5_FACE_OBSERVATION_REVIEW_SCOPES: dict[str, str] = {
    "unknown_cell": "cell",
    "conflict_cell": "cell",
    "unsupported_slot": "slot",
    "degenerate_projection": "claim",
    "no_source_pixels": "claim",
    "off_frame": "claim",
    "empty_coverage": "claim",
}

# ---------------------------------------------------------------------------
# Digest and canonical-JSON helpers (shared canonicalization with V3)
# ---------------------------------------------------------------------------

_canonical_json = pv._canonical_json
_json_value = pv._json_value
_sha256 = pv._sha256
_hex64 = pv._hex64


def _readonly(array: np.ndarray) -> np.ndarray:
    """A private, C-contiguous copy that no caller can make writeable again.

    Mirrors :func:`face_observation._readonly`: the copy is backed by immutable
    ``bytes``, so ``setflags(write=True)`` cannot reopen it directly or through
    its base.
    """
    copy = np.array(array, copy=True, order="C")
    frozen = np.frombuffer(copy.tobytes(order="C"), dtype=copy.dtype).reshape(copy.shape)
    frozen.setflags(write=False)
    return frozen


def _array_record(array: np.ndarray | None) -> dict | None:
    if array is None:
        return None
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": str(contiguous.dtype),
        "shape": [int(value) for value in contiguous.shape],
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _pixels_sha256(image: np.ndarray) -> str:
    header = {
        "schema": "m5-frame-pixels-1",
        "dtype": "float32",
        "image_shape": [int(image.shape[0]), int(image.shape[1]), 3],
    }
    return hashlib.sha256(
        _canonical_json(header).encode("ascii") + b"\0"
        + np.ascontiguousarray(image, dtype=np.float32).tobytes(order="C")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Six slots from outward world-normal geometry
# ---------------------------------------------------------------------------

def _slot_table() -> dict[tuple[int, int], str]:
    table: dict[tuple[int, int], str] = {}
    for slot, basis in FACE_BASIS.items():
        normal = basis[0]
        axis = next(index for index, value in enumerate(normal) if float(value) != 0.0)
        sign = 1 if float(normal[axis]) > 0.0 else -1
        table[(axis, sign)] = slot
    return table


SLOT_OF_NORMAL: dict[tuple[int, int], str] = _slot_table()
"""Canonical outward-normal ``(axis, sign)`` to world-face slot."""
assert set(SLOT_OF_NORMAL) == {(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)}
assert set(SLOT_OF_NORMAL.values()) == set(FACE_SLOTS)


def slot_of_normal(axis: int, outward_sign: int) -> str:
    """The single world-face slot whose canonical normal is ``sign * e_axis``."""
    if type(axis) is not int or axis not in (0, 1, 2):
        raise ValueError("world normal axis must be 0, 1 or 2")
    if outward_sign not in (-1, 1) or isinstance(outward_sign, bool):
        raise ValueError("outward sign must be exactly -1 or +1")
    return SLOT_OF_NORMAL[(axis, int(outward_sign))]


# ---------------------------------------------------------------------------
# Per-frame / per-reading rectification context
# ---------------------------------------------------------------------------

def _rectification_payload(rectification: lattice_planes.PlaneRectification) -> dict:
    return {
        "axis_matrix": [list(row) for row in rectification.axis_matrix],
        "focal": float(rectification.focal),
        "principal_point": [float(value) for value in rectification.principal_point],
        "centre": [float(value) for value in rectification.centre],
    }


@dataclass(frozen=True, slots=True)
class ReadingRectification:
    """One original reading's typed image-rectification context."""

    reading_axis: int
    rectification: lattice_planes.PlaneRectification
    rectification_sha256: str

    def __post_init__(self) -> None:
        if type(self.reading_axis) is not int or self.reading_axis not in (0, 1, 2):
            raise ValueError("rectification reading axis is invalid")
        if not isinstance(self.rectification, lattice_planes.PlaneRectification):
            raise ValueError("rectification must be a PlaneRectification record")
        if (not _hex64(self.rectification_sha256)
                or self.rectification_sha256 != _sha256(self._payload())):
            raise ValueError("rectification digest mismatch")

    def _payload(self) -> dict:
        return {
            "reading_axis": self.reading_axis,
            "rectification": _rectification_payload(self.rectification),
        }

    @classmethod
    def build(cls, *, reading_axis: int,
              rectification: lattice_planes.PlaneRectification) -> "ReadingRectification":
        provisional = object.__new__(cls)
        object.__setattr__(provisional, "reading_axis", reading_axis)
        object.__setattr__(provisional, "rectification", rectification)
        object.__setattr__(provisional, "rectification_sha256", "")
        object.__setattr__(provisional, "rectification_sha256",
                           _sha256(provisional._payload()))
        cls.__post_init__(provisional)
        return provisional

    def to_record(self) -> dict:
        record = self._payload()
        record["rectification_sha256"] = self.rectification_sha256
        return record


def _frame_context_payload(value: "FrameRectificationContext") -> dict:
    return {
        "frame_id": value.frame_id,
        "image_sha256": value.image_sha256,
        "pixels_sha256": value.pixels_sha256,
        "image_shape": [int(value.image_shape[0]), int(value.image_shape[1])],
        "usable_mask_sha256": value.usable_mask_sha256,
        "rectifications": [item.to_record() for item in value.rectifications],
    }


@dataclass(frozen=True, eq=False, slots=True)
class FrameRectificationContext:
    """One frame's immutable image, usable mask and per-reading calibration.

    ``image_sha256`` is the *upstream* image digest the original supports carry;
    it is an opaque accepted record, never recomputed here.  ``pixels_sha256`` is
    recomputed from the retained float32 pixels so the exact consumed bytes are
    identified, and ``usable_mask_sha256`` is recomputed with V1's own
    :func:`plane_region_support.usable_mask_sha256`.
    """

    frame_id: str
    image: np.ndarray
    """``(height, width, 3)`` float32, retained as a read-only copy."""
    image_sha256: str
    """Upstream image digest the original supports declare (opaque, not recomputed)."""
    pixels_sha256: str
    """Digest of the retained pixels, recomputed here."""
    image_shape: tuple[int, int]
    usable_mask: np.ndarray
    """``(height, width)`` bool, retained as a read-only copy."""
    usable_mask_sha256: str
    rectifications: tuple[ReadingRectification, ...]
    context_sha256: str

    def __post_init__(self) -> None:
        if type(self.frame_id) is not str or not self.frame_id:
            raise ValueError("frame context id must be a non-empty string")
        if not _hex64(self.image_sha256):
            raise ValueError("frame context image digest is malformed")
        if (not isinstance(self.image, np.ndarray) or self.image.ndim != 3
                or self.image.shape[2] != 3 or self.image.dtype != np.dtype(np.float32)
                or 0 in self.image.shape[:2]):
            raise ValueError("frame context image must be a non-empty (h, w, 3) float32 array")
        if not bool(np.isfinite(self.image).all()):
            raise ValueError("frame context image must be finite")
        if (not isinstance(self.usable_mask, np.ndarray)
                or self.usable_mask.dtype != np.dtype(bool)
                or self.usable_mask.ndim != 2 or self.usable_mask.size == 0):
            raise ValueError("frame context usable mask must be a non-empty 2D boolean array")
        if tuple(self.usable_mask.shape) != tuple(self.image.shape[:2]):
            raise ValueError("frame context usable mask must match the image dimensions")
        if (not isinstance(self.image_shape, tuple) or len(self.image_shape) != 2
                or any(type(value) is not int or value <= 0 for value in self.image_shape)
                or self.image_shape
                != (int(self.image.shape[0]), int(self.image.shape[1]))):
            raise ValueError("frame context image shape is invalid")
        if (not isinstance(self.rectifications, tuple) or not self.rectifications
                or any(not isinstance(item, ReadingRectification)
                       for item in self.rectifications)):
            raise ValueError("frame context rectifications are malformed")
        axes = [item.reading_axis for item in self.rectifications]
        if axes != sorted(set(axes)):
            raise ValueError("frame context rectifications must be unique and axis-ordered")
        if self.pixels_sha256 != _pixels_sha256(self.image):
            raise ValueError("frame context pixels digest does not replay")
        if self.usable_mask_sha256 != v1.usable_mask_sha256(self.usable_mask):
            raise ValueError("frame context usable mask digest does not replay")
        if (not _hex64(self.context_sha256)
                or self.context_sha256 != _sha256(_frame_context_payload(self))):
            raise ValueError("frame context digest mismatch")

    @classmethod
    def build(cls, *, frame_id: str, image: Any, image_sha256: str, usable_mask: Any,
              rectifications: Sequence[ReadingRectification]) -> "FrameRectificationContext":
        """Validate, copy and digest one frame's immutable rectification context."""
        if type(frame_id) is not str or not frame_id:
            raise ValueError("frame context id must be a non-empty string")
        if not _hex64(image_sha256):
            raise ValueError("frame context image digest is malformed")
        if not isinstance(rectifications, Sequence) or isinstance(rectifications, (str, bytes)):
            raise ValueError("frame context rectifications must be a sequence")
        prepared: list[ReadingRectification] = []
        for item in rectifications:
            if not isinstance(item, ReadingRectification):
                raise ValueError("frame context rectifications must be typed records")
            prepared.append(item)
        prepared.sort(key=lambda item: item.reading_axis)
        raw_image = np.asarray(image)
        if (raw_image.ndim != 3 or raw_image.shape[2] != 3 or raw_image.dtype.kind not in "uif"
                or raw_image.dtype.kind == "b" or 0 in raw_image.shape[:2]):
            raise ValueError("frame context image must be a non-empty (h, w, 3) numeric array")
        retained_image = _readonly(np.asarray(raw_image, dtype=np.float32))
        if not bool(np.isfinite(retained_image).all()):
            raise ValueError("frame context image must be finite")
        raw_mask = np.asarray(usable_mask)
        if raw_mask.dtype != np.dtype(bool) or raw_mask.ndim != 2 or raw_mask.size == 0:
            raise ValueError("frame context usable mask must be a non-empty 2D boolean array")
        retained_mask = _readonly(raw_mask)
        if tuple(retained_mask.shape) != tuple(retained_image.shape[:2]):
            raise ValueError("frame context usable mask must match the image dimensions")
        provisional = object.__new__(cls)
        object.__setattr__(provisional, "frame_id", frame_id)
        object.__setattr__(provisional, "image", retained_image)
        object.__setattr__(provisional, "image_sha256", image_sha256)
        object.__setattr__(provisional, "pixels_sha256", _pixels_sha256(retained_image))
        object.__setattr__(provisional, "image_shape",
                           (int(retained_image.shape[0]), int(retained_image.shape[1])))
        object.__setattr__(provisional, "usable_mask", retained_mask)
        object.__setattr__(provisional, "usable_mask_sha256",
                           v1.usable_mask_sha256(retained_mask))
        object.__setattr__(provisional, "rectifications", tuple(prepared))
        object.__setattr__(provisional, "context_sha256", "")
        object.__setattr__(provisional, "context_sha256",
                           _sha256(_frame_context_payload(provisional)))
        cls.__post_init__(provisional)
        return provisional

    def rectification_for(self, reading_axis: int) -> ReadingRectification | None:
        """The typed rectification context of one reading axis, or ``None``."""
        for item in self.rectifications:
            if item.reading_axis == reading_axis:
                return item
        return None

    def to_record(self) -> dict:
        record = _frame_context_payload(self)
        record["context_sha256"] = self.context_sha256
        return record

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FrameRectificationContext):
            return NotImplemented
        return (
            self.frame_id == other.frame_id
            and self.image_sha256 == other.image_sha256
            and self.pixels_sha256 == other.pixels_sha256
            and self.usable_mask_sha256 == other.usable_mask_sha256
            and self.context_sha256 == other.context_sha256
            and np.array_equal(self.image, other.image)
            and np.array_equal(self.usable_mask, other.usable_mask)
        )

    def __hash__(self) -> int:
        return hash(self.context_sha256)


def frame_context_from_readings(
    *,
    frame_id: str,
    image: Any,
    image_sha256: str,
    usable_mask: Any,
    readings: Sequence[lattice_planes.PlaneReading],
) -> FrameRectificationContext:
    """Derive one frame context from the original ``PlaneReading`` records.

    This is the only sanctioned way to obtain the per-reading
    :class:`~lattice_planes.PlaneRectification` context: it is copied straight out
    of the original accepted readings, never invented.  The bridge still
    re-derives each rectification from the original universe before use, so a
    hand-built context that merely *claims* this provenance is refused.
    """
    if not isinstance(readings, Sequence) or isinstance(readings, (str, bytes)) or not readings:
        raise ValueError("readings must be a non-empty sequence of original PlaneReading records")
    rectifications: list[ReadingRectification] = []
    for reading in readings:
        if not isinstance(reading, lattice_planes.PlaneReading):
            raise ValueError("readings must be original PlaneReading records")
        if reading.frame != frame_id:
            raise ValueError("reading frame disagrees with the frame context")
        rectifications.append(ReadingRectification.build(
            reading_axis=int(reading.axis), rectification=reading.rectification))
    return FrameRectificationContext.build(
        frame_id=frame_id, image=image, image_sha256=image_sha256,
        usable_mask=usable_mask, rectifications=tuple(rectifications))


# ---------------------------------------------------------------------------
# Budget, work and refusal
# ---------------------------------------------------------------------------

_BUDGET_FIELDS: tuple[str, ...] = (
    "max_claims",
    "max_frames",
    "max_rectified_pixels",
    "max_payload_bytes",
    "max_replay_checks",
)
_WORK_FIELDS: tuple[str, ...] = (
    "frames",
    "claims",
    "observations",
    "reviews",
    "rectified_pixels",
    "payload_bytes",
    "replay_checks",
)
_WORK_TO_CAP: dict[str, str] = {
    "frames": "max_frames",
    "claims": "max_claims",
    "rectified_pixels": "max_rectified_pixels",
    "payload_bytes": "max_payload_bytes",
    "replay_checks": "max_replay_checks",
}
_CAP_TO_WORK = {cap: work for work, cap in _WORK_TO_CAP.items()}
_TICKED_WORK: tuple[str, ...] = _WORK_FIELDS
_BUDGET_LIKE_REASONS = ("budget_exhausted",)


@dataclass(frozen=True, slots=True)
class PlaneFaceObservationBudget:
    """Explicit positive whole-request limits for the bridge.  No default."""

    max_claims: int
    max_frames: int
    max_rectified_pixels: int
    max_payload_bytes: int
    max_replay_checks: int
    budget_sha256: str

    def __post_init__(self) -> None:
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 1
               for name in _BUDGET_FIELDS):
            raise ValueError("face observation budget limits must be positive integers")
        if (not _hex64(self.budget_sha256)
                or self.budget_sha256 != _sha256(self._payload())):
            raise ValueError("face observation budget digest mismatch")

    def _payload(self) -> dict:
        return {name: getattr(self, name) for name in _BUDGET_FIELDS}

    def to_record(self) -> dict:
        record = self._payload()
        record["budget_sha256"] = self.budget_sha256
        return record


def make_face_observation_budget(**caps: int) -> PlaneFaceObservationBudget:
    """Build a self-validating bridge budget from explicit caps (no defaults)."""
    payload: dict[str, int] = {}
    for name in _BUDGET_FIELDS:
        if name not in caps:
            raise ValueError("face observation budget is missing a limit")
        value = caps[name]
        if type(value) is not int or value < 1:
            raise ValueError("face observation budget limits must be positive integers")
        payload[name] = value
    return PlaneFaceObservationBudget(**payload, budget_sha256=_sha256(payload))


@dataclass(frozen=True, slots=True)
class PlaneFaceObservationWork:
    """Exact observed bridge work counters."""

    frames: int
    claims: int
    observations: int
    reviews: int
    rectified_pixels: int
    payload_bytes: int
    replay_checks: int
    work_sha256: str

    def __post_init__(self) -> None:
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 0
               for name in _WORK_FIELDS):
            raise ValueError("face observation work counters must be non-negative integers")
        if self.observations > self.claims:
            raise ValueError("emitted observations cannot exceed attempted claims")
        if (not _hex64(self.work_sha256)
                or self.work_sha256 != _sha256(self._payload())):
            raise ValueError("face observation work digest mismatch")

    def _payload(self) -> dict:
        return {name: getattr(self, name) for name in _WORK_FIELDS}

    def to_record(self) -> dict:
        record = self._payload()
        record["work_sha256"] = self.work_sha256
        return record


def _make_work(counters: Mapping[str, int]) -> PlaneFaceObservationWork:
    payload: dict[str, int] = {}
    for name in _WORK_FIELDS:
        value = counters.get(name, 0)
        if type(value) is not int or value < 0:
            raise ValueError("face observation work counters must be non-negative integers")
        payload[name] = value
    return PlaneFaceObservationWork(**payload, work_sha256=_sha256(payload))


@dataclass(frozen=True, slots=True)
class PlaneFaceObservationRefusal:
    """One typed, atomic bridge refusal.  It carries no partial observations."""

    reason: str
    detail: dict
    detail_sha256: str
    work: PlaneFaceObservationWork | None
    refusal_sha256: str

    def __post_init__(self) -> None:
        if self.reason not in M5_FACE_OBSERVATION_REFUSAL_REASONS:
            raise ValueError("face observation refusal reason is not in the closed enum")
        normalized = _json_value(self.detail)
        if not isinstance(normalized, dict):
            raise ValueError("face observation refusal detail must be a JSON object")
        encoded = _canonical_json(normalized)
        if self.detail_sha256 != hashlib.sha256(encoded.encode("utf-8")).hexdigest():
            raise ValueError("face observation refusal detail digest mismatch")
        if self.work is not None and not isinstance(self.work, PlaneFaceObservationWork):
            raise ValueError("face observation refusal work is malformed")
        if set(normalized) & {"observations", "cells", "patches", "reviews"}:
            raise ValueError("face observation refusal must not embed partial records")
        if self.reason in _BUDGET_LIKE_REASONS:
            if self.work is None or set(normalized) != {"cap", "limit", "observed"}:
                raise ValueError("face observation budget refusal is malformed")
            cap = normalized["cap"]
            limit = normalized["limit"]
            observed = normalized["observed"]
            if (cap not in _CAP_TO_WORK or type(limit) is not int
                    or type(observed) is not int or limit < 1 or observed <= limit):
                raise ValueError("face observation budget refusal evidence is malformed")
            if getattr(self.work, _CAP_TO_WORK[cap]) != observed:
                raise ValueError(
                    "face observation budget refusal disagrees with its work counter")
        elif self.work is not None:
            raise ValueError("non-budget face observation refusal must not carry work")
        payload = {
            "reason": self.reason,
            "detail_sha256": self.detail_sha256,
            "work": None if self.work is None else self.work.to_record(),
        }
        if (not _hex64(self.refusal_sha256)
                or self.refusal_sha256 != _sha256(payload)):
            raise ValueError("face observation refusal digest mismatch")
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
                  work: PlaneFaceObservationWork | None = None
                  ) -> PlaneFaceObservationRefusal:
    if reason not in M5_FACE_OBSERVATION_REFUSAL_REASONS:
        reason = "invalid_input"
    normalized = _json_value({} if detail is None else detail)
    detail_sha = hashlib.sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()
    payload = {
        "reason": reason,
        "detail_sha256": detail_sha,
        "work": None if work is None else work.to_record(),
    }
    return PlaneFaceObservationRefusal(reason, normalized, detail_sha, work,
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
    """Truthful, checked bridge work counters with stop-on-cap behaviour."""

    __slots__ = ("budget", "frames", "claims", "observations", "reviews",
                 "rectified_pixels", "payload_bytes", "replay_checks")

    def __init__(self, budget: PlaneFaceObservationBudget):
        self.budget = budget
        for name in _WORK_FIELDS:
            setattr(self, name, 0)

    def tick(self, field: str, amount: int = 1) -> None:
        if field not in _TICKED_WORK or type(amount) is not int or amount < 1:
            raise ValueError("invalid face observation counter update")
        setattr(self, field, getattr(self, field) + amount)
        for work_name, cap_name in _WORK_TO_CAP.items():
            observed = getattr(self, work_name)
            limit = getattr(self.budget, cap_name)
            if observed > limit:
                raise _BudgetStop(cap_name, limit, observed)

    def snapshot(self) -> PlaneFaceObservationWork:
        return _make_work({name: getattr(self, name) for name in _WORK_FIELDS})


class _RecordingCounter:
    """Unlimited work recorder using the exact ``tick`` calls of ``_Counter``.

    The authoritative record set is derived once with this recorder, so the
    caller's budget can be applied afterwards by replaying the recorded events
    through a real :class:`_Counter`.  That is what makes every invalid,
    provenance, replay and context decision precede any budget decision.
    """

    __slots__ = ("events", "counts")

    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.counts: dict[str, int] = {name: 0 for name in _WORK_FIELDS}

    def tick(self, field: str, amount: int = 1) -> None:
        if field not in _TICKED_WORK or type(amount) is not int or amount < 1:
            raise ValueError("invalid face observation counter update")
        self.events.append(("tick", field, amount))
        self.counts[field] += amount

    def snapshot(self) -> PlaneFaceObservationWork:
        return _make_work(self.counts)


# ---------------------------------------------------------------------------
# Bound claim/observation/review records
# ---------------------------------------------------------------------------

def _binding_payload(value: "FaceObservationBinding") -> dict:
    return {
        "cell": list(value.cell),
        "slot": value.slot,
        "frame_id": value.frame_id,
        "reading_axis": value.reading_axis,
        "world_normal_axis": value.world_normal_axis,
        "world_normal_sign": value.world_normal_sign,
        "outward_sign": value.outward_sign,
        "row": value.row,
        "column": value.column,
        "claim_sha256": value.claim_sha256,
        "reading_identity_sha256": value.reading_identity_sha256,
        "support_sha256": value.support_sha256,
        "face_sha256": value.face_sha256,
        "frame_context_sha256": value.frame_context_sha256,
        "rectification_sha256": value.rectification_sha256,
        "image_pixels_sha256": value.image_pixels_sha256,
        "usable_mask_sha256": value.usable_mask_sha256,
        "observation_sha256": value.observation_sha256,
    }


@dataclass(frozen=True, slots=True)
class FaceObservationBinding:
    """One emitted observation's exact claim, context and digest binding."""

    cell: tuple[int, int, int]
    slot: str
    frame_id: str
    reading_axis: int
    world_normal_axis: int
    world_normal_sign: int
    outward_sign: int
    row: int
    column: int
    claim_sha256: str
    reading_identity_sha256: str
    support_sha256: str
    face_sha256: str
    frame_context_sha256: str
    rectification_sha256: str
    image_pixels_sha256: str
    usable_mask_sha256: str
    observation_sha256: str
    binding_sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(self.cell, tuple) or len(self.cell) != 3
                or any(type(value) is not int for value in self.cell)):
            raise ValueError("binding cell must be three integers")
        if self.slot not in FACE_SLOTS:
            raise ValueError("binding slot is not a canonical world-face slot")
        if type(self.frame_id) is not str or not self.frame_id:
            raise ValueError("binding frame id is invalid")
        if type(self.reading_axis) is not int or self.reading_axis not in (0, 1, 2):
            raise ValueError("binding reading axis is invalid")
        if type(self.world_normal_axis) is not int or self.world_normal_axis not in (0, 1, 2):
            raise ValueError("binding world normal axis is invalid")
        if self.world_normal_sign not in (-1, 1) or self.outward_sign not in (-1, 1):
            raise ValueError("binding signs are invalid")
        try:
            expected_slot = slot_of_normal(self.world_normal_axis, self.outward_sign)
        except ValueError as error:
            raise ValueError("binding outward normal is malformed") from error
        if expected_slot != self.slot:
            raise ValueError("binding slot disagrees with its outward normal geometry")
        if type(self.row) is not int or self.row < 0 \
                or type(self.column) is not int or self.column < 0:
            raise ValueError("binding cell indices are invalid")
        for name in ("claim_sha256", "reading_identity_sha256", "support_sha256",
                     "face_sha256", "frame_context_sha256", "rectification_sha256",
                     "image_pixels_sha256", "usable_mask_sha256",
                     "observation_sha256", "binding_sha256"):
            if not _hex64(getattr(self, name)):
                raise ValueError("binding digests are malformed")
        if self.binding_sha256 != _sha256(_binding_payload(self)):
            raise ValueError("binding digest mismatch")

    @classmethod
    def build(cls, **fields: Any) -> "FaceObservationBinding":
        provisional = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "binding_sha256", "")
        object.__setattr__(provisional, "binding_sha256",
                           _sha256(_binding_payload(provisional)))
        cls.__post_init__(provisional)
        return provisional

    def to_record(self) -> dict:
        record = _binding_payload(self)
        record["binding_sha256"] = self.binding_sha256
        return record


def _review_payload(value: "FaceObservationReview") -> dict:
    return {
        "reason": value.reason,
        "scope": value.scope,
        "coordinate": None if value.coordinate is None else list(value.coordinate),
        "slot": value.slot,
        "frame_ids": list(value.frame_ids),
        "claim_sha256s": list(value.claim_sha256s),
        "support_sha256s": list(value.support_sha256s),
        "detail": dict(value.detail),
    }


@dataclass(frozen=True, slots=True)
class FaceObservationReview:
    """One typed omission of a cell, slot or claim from the observation set.

    An unknown V3 cell, an explicitly conflicting V3 cell, a cell/slot pair with
    no claim, and a claim whose quad cannot be rectified into any usable coverage
    are all recorded here instead of being silently dropped or emitted as a
    partial observation.
    """

    reason: str
    scope: str
    coordinate: tuple[int, int, int] | None
    slot: str | None
    frame_ids: tuple[str, ...]
    claim_sha256s: tuple[str, ...]
    support_sha256s: tuple[str, ...]
    detail: dict
    review_sha256: str

    def __post_init__(self) -> None:
        if self.reason not in M5_FACE_OBSERVATION_REVIEW_REASONS:
            raise ValueError("review reason is not in the closed enum")
        if self.scope != M5_FACE_OBSERVATION_REVIEW_SCOPES[self.reason]:
            raise ValueError("review scope disagrees with its reason")
        if self.coordinate is not None and (
                not isinstance(self.coordinate, tuple) or len(self.coordinate) != 3
                or any(type(value) is not int for value in self.coordinate)):
            raise ValueError("review coordinate must be three integers or None")
        if self.slot is not None and self.slot not in FACE_SLOTS:
            raise ValueError("review slot is not a canonical world-face slot")
        if self.scope == "slot" and (self.coordinate is None or self.slot is None):
            raise ValueError("slot reviews require a coordinate and a slot")
        if self.scope == "claim":
            if self.coordinate is None or self.slot is None or not self.frame_ids \
                    or not self.claim_sha256s:
                raise ValueError("claim reviews require a coordinate, slot, frame and claim")
        for name in ("frame_ids", "claim_sha256s", "support_sha256s"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(set(values)) != len(values):
                raise ValueError("review reference tuples must be unique")
            if any(type(value) is not str or not value for value in values):
                raise ValueError("review references must be non-empty strings")
        for value in self.claim_sha256s + self.support_sha256s:
            if not _hex64(value):
                raise ValueError("review claim/support references must be digests")
        normalized = _json_value(self.detail)
        if not isinstance(normalized, dict):
            raise ValueError("review detail must be a JSON object")
        if (not _hex64(self.review_sha256)
                or self.review_sha256 != _sha256(_review_payload(self))):
            raise ValueError("review digest mismatch")
        object.__setattr__(self, "detail", normalized)

    @classmethod
    def build(cls, **fields: Any) -> "FaceObservationReview":
        provisional = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "review_sha256", "")
        object.__setattr__(provisional, "review_sha256",
                           _sha256(_review_payload(provisional)))
        cls.__post_init__(provisional)
        return provisional

    def to_record(self) -> dict:
        record = _review_payload(self)
        record["review_sha256"] = self.review_sha256
        return record


def _observation_payload(value: FaceObservation) -> dict:
    return {
        "cell": list(value.cell),
        "slot": value.slot,
        "frame_id": value.frame_id,
        "normal": [float(component) for component in value.normal],
        "usable_coverage": float(value.usable_coverage),
        "support_frames": list(value.support_frames),
        "provenance": list(value.provenance),
        "weight": float(value.weight),
        "patch": _array_record(value.patch),
        "valid": _array_record(value.valid),
        "coverage": _array_record(value.coverage),
    }


def face_observation_sha256(observation: FaceObservation) -> str:
    """Content digest of one emitted observation, including its retained arrays."""
    if not isinstance(observation, FaceObservation):
        raise ValueError("face_observation_sha256 requires a FaceObservation")
    try:
        return _sha256(_observation_payload(observation))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("observation is not digestible") from error


def _observation_payload_bytes(observation: FaceObservation) -> int:
    total = int(observation.patch.nbytes)
    if observation.valid is not None:
        total += int(observation.valid.nbytes)
    if observation.coverage is not None:
        total += int(observation.coverage.nbytes)
    return total


def face_observation_sort_key(observation: FaceObservation) -> tuple:
    """Deterministic ``(cell, slot, frame)`` identity order."""
    return (tuple(observation.cell), FACE_SLOTS.index(observation.slot),
            observation.frame_id)


def _binding_sort_key(binding: FaceObservationBinding) -> tuple:
    return (tuple(binding.cell), FACE_SLOTS.index(binding.slot), binding.frame_id)


def _review_sort_key(review: FaceObservationReview) -> tuple:
    slot_index = -1 if review.slot is None else FACE_SLOTS.index(review.slot)
    coordinate = (-1, -1, -1) if review.coordinate is None else tuple(review.coordinate)
    return (review.scope, review.reason, coordinate, slot_index,
            review.frame_ids, review.claim_sha256s, review.review_sha256)


def canonical_face_observation_order(records: Sequence[Any]) -> tuple:
    """Canonicalize a reordered input sequence of bridge records."""
    if not isinstance(records, Sequence):
        raise ValueError("face observation records must be a sequence")
    return tuple(sorted(records, key=_record_sort_key))


def _record_sort_key(record: Any) -> tuple:
    if isinstance(record, FaceObservation):
        return ("observation",) + face_observation_sort_key(record)
    if isinstance(record, FaceObservationBinding):
        return ("binding",) + _binding_sort_key(record)
    if isinstance(record, FaceObservationReview):
        return ("review",) + _review_sort_key(record)
    if isinstance(record, PlaneFaceObservationRefusal):
        return ("refusal", record.reason, record.detail_sha256, record.refusal_sha256)
    raise ValueError("face observation sort key requires a bridge record")


# ---------------------------------------------------------------------------
# Canonical result
# ---------------------------------------------------------------------------

_FRAME_BINDING_KEYS = frozenset({
    "frame_id", "image_sha256", "pixels_sha256", "usable_mask_sha256",
    "image_shape", "reading_axes", "context_sha256",
})


def _frame_binding(context: FrameRectificationContext, reading_axes: Sequence[int]) -> dict:
    return {
        "frame_id": context.frame_id,
        "image_sha256": context.image_sha256,
        "pixels_sha256": context.pixels_sha256,
        "usable_mask_sha256": context.usable_mask_sha256,
        "image_shape": [int(context.image_shape[0]), int(context.image_shape[1])],
        "reading_axes": [int(axis) for axis in reading_axes],
        "context_sha256": context.context_sha256,
    }


def _frame_binding_sort_key(binding: Mapping[str, Any]) -> tuple:
    return (binding["frame_id"],)


def _result_payload(value: "PlaneFaceObservationResult") -> dict:
    return {
        "schema_version": value.schema_version,
        "producer_version": value.producer_version,
        "coverage_policy": value.coverage_policy,
        "volume_sha256": value.volume_sha256,
        "placement_sha256": value.placement_sha256,
        "frame_bindings": [dict(binding) for binding in value.frame_bindings],
        "observations": [_observation_payload(item) for item in value.observations],
        "bindings": [item.to_record() for item in value.bindings],
        "reviews": [item.to_record() for item in value.reviews],
        "work": value.work.to_record(),
    }


def _derived_work(value: "PlaneFaceObservationResult") -> dict:
    claim_reviews = sum(1 for review in value.reviews if review.scope == "claim")
    warped_reviews = sum(1 for review in value.reviews
                         if review.reason == "empty_coverage")
    return {
        "frames": len(value.frame_bindings),
        "claims": len(value.observations) + claim_reviews,
        "observations": len(value.observations),
        "reviews": len(value.reviews),
        "rectified_pixels": int(face_sampling.PATCH_SIZE) ** 2 * (
            len(value.observations) + warped_reviews),
        "payload_bytes": sum(_observation_payload_bytes(item)
                             for item in value.observations),
        "replay_checks": (
            len(value.frame_bindings)
            + sum(len(binding["reading_axes"]) for binding in value.frame_bindings)
            + len({binding.support_sha256 for binding in value.bindings})),
    }


@dataclass(frozen=True, slots=True)
class PlaneFaceObservationResult:
    """The frozen, versioned, source-bound plane-homography observation set."""

    schema_version: str
    producer_version: str
    coverage_policy: str
    volume_sha256: str
    placement_sha256: str
    frame_bindings: tuple[dict, ...]
    observations: tuple[FaceObservation, ...]
    bindings: tuple[FaceObservationBinding, ...]
    reviews: tuple[FaceObservationReview, ...]
    work: PlaneFaceObservationWork
    result_sha256: str

    def __post_init__(self) -> None:
        if (self.schema_version != M5_FACE_OBSERVATION_SCHEMA
                or self.producer_version != M5_FACE_OBSERVATION_PRODUCER_VERSION
                or self.coverage_policy != M5_FACE_OBSERVATION_COVERAGE_POLICY):
            raise ValueError("face observation record version or policy is unsupported")
        if not _hex64(self.volume_sha256) or not _hex64(self.placement_sha256):
            raise ValueError("face observation source bindings are malformed")
        if (not isinstance(self.frame_bindings, tuple) or not self.frame_bindings
                or any(not isinstance(binding, dict) for binding in self.frame_bindings)):
            raise ValueError("face observation frame bindings are malformed")
        if tuple(sorted(self.frame_bindings, key=_frame_binding_sort_key)) \
                != self.frame_bindings:
            raise ValueError("face observation frame bindings are not in canonical order")
        frame_ids = [binding["frame_id"] for binding in self.frame_bindings]
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("face observation frame bindings are not unique")
        frames_by_id: dict[str, dict] = {}
        for binding in self.frame_bindings:
            if set(binding) != _FRAME_BINDING_KEYS:
                raise ValueError("face observation frame binding fields are malformed")
            if type(binding["frame_id"]) is not str or not binding["frame_id"]:
                raise ValueError("face observation frame binding id is invalid")
            if any(not _hex64(binding[name]) for name in
                   ("image_sha256", "pixels_sha256", "usable_mask_sha256",
                    "context_sha256")):
                raise ValueError("face observation frame binding digests are malformed")
            shape = binding["image_shape"]
            if (not isinstance(shape, list) or len(shape) != 2
                    or any(type(value) is not int or value <= 0 for value in shape)):
                raise ValueError("face observation frame binding shape is invalid")
            axes = binding["reading_axes"]
            if (not isinstance(axes, list) or not axes
                    or any(type(value) is not int or value not in (0, 1, 2)
                           for value in axes)
                    or axes != sorted(set(axes))):
                raise ValueError("face observation frame binding axes are invalid")
            frames_by_id[binding["frame_id"]] = binding

        if (not isinstance(self.observations, tuple)
                or any(not isinstance(item, FaceObservation)
                       for item in self.observations)):
            raise ValueError("face observation records are malformed")
        identities = [(item.cell, item.slot, item.frame_id) for item in self.observations]
        if len(set(identities)) != len(identities):
            raise ValueError("face observations are not unique")
        expected_order = tuple(sorted(self.observations, key=face_observation_sort_key))
        if expected_order != self.observations:
            raise ValueError("face observations are not in canonical order")

        if (not isinstance(self.bindings, tuple)
                or any(not isinstance(item, FaceObservationBinding)
                       for item in self.bindings)):
            raise ValueError("face observation bindings are malformed")
        if len(self.bindings) != len(self.observations):
            raise ValueError("face observation bindings do not match its observations")
        if any(binding.binding_sha256 != _sha256(_binding_payload(binding))
               for binding in self.bindings):
            raise ValueError("face observation binding digest does not replay")
        frames_by_identity: dict[tuple, set[str]] = {}
        supported_supports: set[str] = set()
        for binding in self.bindings:
            key = (binding.cell, binding.slot)
            frames_by_identity.setdefault(key, set()).add(binding.frame_id)
            supported_supports.add(binding.support_sha256)
            frame = frames_by_id.get(binding.frame_id)
            if frame is None:
                raise ValueError("face observation binding has no frame binding")
            if binding.frame_context_sha256 != frame["context_sha256"]:
                raise ValueError("face observation binding disagrees with its frame context")
            if binding.image_pixels_sha256 != frame["pixels_sha256"]:
                raise ValueError("face observation binding disagrees with its frame pixels")
            if binding.usable_mask_sha256 != frame["usable_mask_sha256"]:
                raise ValueError("face observation binding disagrees with its usable mask")
        for binding, observation in zip(self.bindings, self.observations):
            if (binding.cell, binding.slot, binding.frame_id) \
                    != (observation.cell, observation.slot, observation.frame_id):
                raise ValueError("face observation bindings are not aligned with observations")
            if binding.observation_sha256 != face_observation_sha256(observation):
                raise ValueError("face observation binding digest does not replay")
            if set(observation.support_frames) != frames_by_identity[
                    (observation.cell, observation.slot)]:
                raise ValueError(
                    "support_frames must be exactly the claims of the same cell and slot")
            if (observation.usable_coverage <= 0.0
                    or not observation.has_valid_pixels
                    or not observation.has_effective_coverage):
                raise ValueError("emitted observations must carry usable evidence")

        if (not isinstance(self.reviews, tuple)
                or any(not isinstance(item, FaceObservationReview)
                       for item in self.reviews)):
            raise ValueError("face observation reviews are malformed")
        if tuple(sorted(self.reviews, key=_review_sort_key)) != self.reviews:
            raise ValueError("face observation reviews are not in canonical order")
        review_digests = [review.review_sha256 for review in self.reviews]
        if len(set(review_digests)) != len(review_digests):
            raise ValueError("face observation reviews are not unique")
        for review in self.reviews:
            if review.review_sha256 != _sha256(_review_payload(review)):
                raise ValueError("face observation review digest does not replay")
        if not isinstance(self.work, PlaneFaceObservationWork):
            raise ValueError("face observation work record is malformed")
        derived = _derived_work(self)
        for name in _WORK_FIELDS:
            if getattr(self.work, name) != derived[name]:
                raise ValueError("face observation work does not replay its records")
        if (not _hex64(self.result_sha256)
                or self.result_sha256 != _sha256(_result_payload(self))):
            raise ValueError("face observation result digest mismatch")

    @classmethod
    def build(cls, **fields: Any) -> "PlaneFaceObservationResult":
        provisional = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "result_sha256", "")
        object.__setattr__(provisional, "result_sha256",
                           _sha256(_result_payload(provisional)))
        cls.__post_init__(provisional)
        return provisional

    def to_record(self) -> dict:
        record = _result_payload(self)
        record["result_sha256"] = self.result_sha256
        return record

    def observations_for(self, cell: tuple[int, int, int],
                         slot: str | None = None) -> tuple[FaceObservation, ...]:
        """The emitted observations of one cell, optionally one slot."""
        return tuple(item for item in self.observations
                     if item.cell == cell and (slot is None or item.slot == slot))

    def reviews_for(self, coordinate: tuple[int, int, int]) -> tuple[FaceObservationReview, ...]:
        return tuple(review for review in self.reviews
                     if review.coordinate == coordinate)


# ---------------------------------------------------------------------------
# Source-bound replay, context binding and record derivation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _BoundSource:
    """One accepted support bound to its original reading, context and masks."""

    support: v1.AlignedRegionSupport
    reading: lattice_planes.PlaneReading
    context: FrameRectificationContext
    region_mask: np.ndarray


def _replay_volume(
    result: Any,
    *,
    universe: Any,
    supports: Any,
    placement: Any,
    placement_budget: Any,
    volume_budget: Any,
) -> pv.PlaneVolumeResult:
    """Rerun accepted V3 from the original records and require an exact replay."""
    if not isinstance(result, pv.PlaneVolumeResult):
        raise _InputRefusal("non_volume_input", {"field": "result"})
    if not isinstance(volume_budget, pv.PlaneVolumeBudget):
        raise _InputRefusal("invalid_input", {"field": "volume_budget"})
    if not isinstance(universe, plane_placement.PlacementUniverse):
        raise _InputRefusal("invalid_input", {"field": "universe"})
    if not isinstance(placement, plane_placement.ResolvedFacePlacement):
        raise _InputRefusal("invalid_input", {"field": "placement"})
    if not isinstance(placement_budget, plane_placement.PlacementBudget):
        raise _InputRefusal("invalid_input", {"field": "placement_budget"})
    if (not isinstance(supports, Sequence) or isinstance(supports, (str, bytes))
            or not supports
            or any(not isinstance(item, v1.AlignedRegionSupport) for item in supports)):
        raise _InputRefusal("invalid_input", {"field": "supports"})
    rebuilt = pv.build_plane_volume(
        universe=universe, supports=supports, placement=placement,
        placement_budget=placement_budget, budget=volume_budget)
    if isinstance(rebuilt, pv.VolumeRefusal):
        raise _InputRefusal("volume_replay_refused", {"reason": rebuilt.reason})
    if not isinstance(rebuilt, pv.PlaneVolumeResult):
        raise _InputRefusal("volume_replay_refused", {"detail": type(rebuilt).__name__})
    try:
        replayed = (rebuilt.volume_sha256 == result.volume_sha256
                    and rebuilt.to_record() == result.to_record())
    except (AttributeError, TypeError, ValueError, IndexError, KeyError):
        raise _InputRefusal("replay_mismatch", {"field": "result"})
    if not replayed:
        raise _InputRefusal("replay_mismatch", {"field": "result"})
    return rebuilt


def _bind_contexts(
    frames: Any,
    authoritative: pv.PlaneVolumeResult,
    universe: plane_placement.PlacementUniverse,
    supports: Sequence[v1.AlignedRegionSupport],
    recorder: Any,
) -> dict[str, _BoundSource]:
    """Bind every frame context and support to the original accepted records."""
    if (not isinstance(frames, Sequence) or isinstance(frames, (str, bytes))
            or not frames):
        raise _InputRefusal("invalid_input", {"field": "frames"})
    contexts: list[FrameRectificationContext] = []
    seen_frames: set[str] = set()
    for context in frames:
        if not isinstance(context, FrameRectificationContext):
            raise _InputRefusal("invalid_input", {"field": "frames"})
        if context.frame_id in seen_frames:
            raise _InputRefusal("invalid_input", {"field": "duplicate_frame"})
        seen_frames.add(context.frame_id)
        contexts.append(context)
    contexts.sort(key=lambda item: item.frame_id)
    by_frame = {context.frame_id: context for context in contexts}

    supports_by_digest: dict[str, v1.AlignedRegionSupport] = {}
    for support in supports:
        if support.support_sha256 in supports_by_digest:
            raise _InputRefusal("provenance_mismatch", {"field": "duplicate_support"})
        supports_by_digest[support.support_sha256] = support
    readings: dict[tuple[str, int], lattice_planes.PlaneReading] = {}
    for reading in universe.readings:
        key = (reading.frame, int(reading.axis))
        if key in readings and readings[key] is not reading:
            raise _InputRefusal("provenance_mismatch", {"field": "reading_identity"})
        readings[key] = reading
    universe_masks: dict[str, Any] = {}
    for entry in universe.usable_masks:
        if (not isinstance(entry, tuple) or len(entry) != 2
                or type(entry[0]) is not str):
            raise _InputRefusal("invalid_input", {"field": "usable_masks"})
        if entry[0] in universe_masks and universe_masks[entry[0]] is not entry[1]:
            raise _InputRefusal("provenance_mismatch", {"field": "usable_masks"})
        universe_masks[entry[0]] = entry[1]

    used_axes_by_frame: dict[str, set[int]] = {}
    for binding in authoritative.support_bindings:
        used_axes_by_frame.setdefault(binding["frame"], set()).add(
            int(binding["reading_axis"]))
    for frame_id in sorted(used_axes_by_frame):
        if frame_id not in by_frame:
            raise _InputRefusal("missing_frame_context", {"frame": frame_id})

    for context in contexts:
        if (not isinstance(context.image, np.ndarray) or context.image.ndim != 3
                or context.image.shape[2] != 3
                or context.image.dtype != np.dtype(np.float32)
                or 0 in context.image.shape[:2]
                or not bool(np.isfinite(context.image).all())):
            raise _InputRefusal("frame_context_mismatch", {
                "field": "image", "frame": context.frame_id})
        if (not isinstance(context.usable_mask, np.ndarray)
                or context.usable_mask.dtype != np.dtype(bool)
                or context.usable_mask.ndim != 2 or context.usable_mask.size == 0
                or tuple(context.usable_mask.shape) != tuple(context.image.shape[:2])):
            raise _InputRefusal("frame_context_mismatch", {
                "field": "usable_mask", "frame": context.frame_id})
        if context.pixels_sha256 != _pixels_sha256(context.image):
            raise _InputRefusal("frame_context_mismatch", {
                "field": "pixels_sha256", "frame": context.frame_id})
        if context.usable_mask_sha256 != v1.usable_mask_sha256(context.usable_mask):
            raise _InputRefusal("frame_context_mismatch", {
                "field": "usable_mask_sha256", "frame": context.frame_id})
        if (not _hex64(context.context_sha256)
                or context.context_sha256 != _sha256(_frame_context_payload(context))):
            raise _InputRefusal("frame_context_mismatch", {
                "field": "context_sha256", "frame": context.frame_id})
        if tuple(context.image_shape) != (int(context.image.shape[0]),
                                          int(context.image.shape[1])):
            raise _InputRefusal("frame_context_mismatch", {
                "field": "image_shape", "frame": context.frame_id})
        used = used_axes_by_frame.get(context.frame_id)
        if used is None:
            raise _InputRefusal("unreferenced_frame", {"frame": context.frame_id})
        supplied = {item.reading_axis for item in context.rectifications}
        if supplied - used:
            raise _InputRefusal("frame_context_mismatch", {
                "field": "unreferenced_rectification", "frame": context.frame_id})
        if used - supplied:
            raise _InputRefusal("missing_rectification", {"frame": context.frame_id})
        recorder.tick("frames")
        recorder.tick("replay_checks")
        for item in context.rectifications:
            reading = readings.get((context.frame_id, item.reading_axis))
            if reading is None:
                raise _InputRefusal("missing_rectification", {
                    "frame": context.frame_id, "reading_axis": item.reading_axis})
            if item.rectification != reading.rectification:
                raise _InputRefusal("frame_context_mismatch", {
                    "field": "rectification", "frame": context.frame_id,
                    "reading_axis": item.reading_axis})
            recorder.tick("replay_checks")

    bound: dict[str, _BoundSource] = {}
    for digest in sorted(supports_by_digest):
        support = supports_by_digest[digest]
        identity = support.reading_identity
        frame_id = identity.frame
        reading_axis = int(identity.reading_axis)
        context = by_frame.get(frame_id)
        if context is None:
            raise _InputRefusal("missing_frame_context", {"frame": frame_id})
        reading = readings.get((frame_id, reading_axis))
        if reading is None:
            raise _InputRefusal("missing_provenance", {"field": "reading"})
        try:
            expected_calibration = v1._calibration_from_reading(reading)
        except (AttributeError, TypeError, ValueError) as error:
            raise _InputRefusal("missing_rectification", {
                "frame": frame_id, "reading_axis": reading_axis}) from error
        if support.calibration != expected_calibration:
            raise _InputRefusal("frame_context_mismatch", {
                "field": "calibration", "support_sha256": digest})
        rectification = context.rectification_for(reading_axis)
        if rectification is None:
            raise _InputRefusal("missing_rectification", {
                "frame": frame_id, "reading_axis": reading_axis})
        if rectification.rectification != reading.rectification:
            raise _InputRefusal("frame_context_mismatch", {
                "field": "rectification", "support_sha256": digest})
        if context.image_sha256 != support.image_sha256:
            raise _InputRefusal("frame_context_mismatch", {
                "field": "image_sha256", "frame": frame_id})
        if context.usable_mask_sha256 != support.usable_mask_sha256:
            raise _InputRefusal("frame_context_mismatch", {
                "field": "usable_mask_sha256", "frame": frame_id})
        original_mask = universe_masks.get(support.reading_identity_sha256)
        if original_mask is None:
            raise _InputRefusal("missing_provenance", {"field": "usable_mask"})
        try:
            original_mask_digest = v1.usable_mask_sha256(original_mask)
        except (TypeError, ValueError) as error:
            raise _InputRefusal("provenance_mismatch", {
                "field": "usable_mask"}) from error
        if original_mask_digest != support.usable_mask_sha256:
            raise _InputRefusal("provenance_mismatch", {"field": "usable_mask"})
        if tuple(context.image_shape) != tuple(support.image_shape):
            raise _InputRefusal("frame_context_mismatch", {
                "field": "image_shape", "frame": frame_id})
        evidence = reading.region_evidence
        if not isinstance(evidence, lattice_planes.PlaneRegionEvidence):
            raise _InputRefusal("missing_provenance", {"field": "region_evidence"})
        try:
            region_mask = lattice_planes._decode_region_evidence(
                evidence, reading.axis, reading.polygon)
        except (TypeError, ValueError) as error:
            raise _InputRefusal("provenance_mismatch", {
                "field": "region_evidence"}) from error
        if (support.region_mask_sha256 != evidence.mask_sha256
                or support.source_region_id != evidence.source_region_id):
            raise _InputRefusal("provenance_mismatch", {"field": "region_mask"})
        recorder.tick("replay_checks")
        bound[digest] = _BoundSource(support, reading, context, region_mask)

    referenced = {binding["support_sha256"] for binding in authoritative.support_bindings}
    if referenced - set(bound):
        raise _InputRefusal("missing_provenance", {"field": "support"})
    return bound


def _claim_review(
    reason: str,
    cell: pv.SupportedCellObservation,
    claim: pv.CellClaim,
    slot: str,
    detail: dict,
) -> FaceObservationReview:
    return FaceObservationReview.build(
        reason=reason, scope="claim", coordinate=cell.coordinate, slot=slot,
        frame_ids=(claim.frame,), claim_sha256s=(claim.claim_sha256,),
        support_sha256s=(claim.support_sha256,), detail=detail)


def _rectify_claim(
    cell: pv.SupportedCellObservation,
    claim: pv.CellClaim,
    slot: str,
    support_frames: tuple[str, ...],
    bound: _BoundSource,
    volume_sha256: str,
    recorder: Any,
) -> FaceObservation | FaceObservationReview:
    """Rebuild one claim's V1 image quad and rectify its usable coverage."""
    axis = int(claim.reading_axis)
    try:
        homography = v1._exact_homography(axis, bound.support.calibration,
                                          bound.support.alternative)
    except (AttributeError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return _claim_review("degenerate_projection", cell, claim, slot,
                             {"stage": "homography"})
    origin_u = Fraction(*bound.support.origin_uv[0])
    origin_v = Fraction(*bound.support.origin_uv[1])
    u_start = origin_u + int(claim.column)
    v_start = origin_v + int(claim.row)
    try:
        projected = []
        for u, v in ((u_start, v_start), (u_start + 1, v_start),
                     (u_start + 1, v_start + 1), (u_start, v_start + 1)):
            point = v1._project(homography, u, v)
            if point is None:
                return _claim_review("degenerate_projection", cell, claim, slot,
                                     {"stage": "projection"})
            projected.append((float(point[0]), float(point[1])))
    except (AttributeError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return _claim_review("degenerate_projection", cell, claim, slot,
                             {"stage": "projection"})
    quad = np.asarray(projected, dtype=np.float32)
    area = face_sampling.quad_area(quad)
    if not np.isfinite(area) or area <= 0.0 or not bool(np.isfinite(quad).all()):
        return _claim_review("degenerate_projection", cell, claim, slot,
                             {"stage": "area"})
    height, width = bound.context.image_shape
    if (float(quad[:, 0].max()) < 0.0 or float(quad[:, 0].min()) > float(width)
            or float(quad[:, 1].max()) < 0.0 or float(quad[:, 1].min()) > float(height)):
        return _claim_review("off_frame", cell, claim, slot,
                             {"image_shape": [int(height), int(width)]})
    source_valid = bound.region_mask & bound.context.usable_mask
    if not bool(source_valid.any()):
        return _claim_review("no_source_pixels", cell, claim, slot,
                             {"source_pixels": 0})
    rectified = face_sampling.rectify_face_valid(
        bound.context.image, quad, source_valid, size=int(face_sampling.PATCH_SIZE))
    if rectified is None:
        raise _InputRefusal("invalid_input", {"field": "rectifier"})
    patch, valid, coverage = rectified
    recorder.tick("rectified_pixels", int(patch.shape[0]) * int(patch.shape[1]))
    coverage = np.clip(np.asarray(coverage, dtype=np.float64), 0.0, 1.0)
    usable = float(coverage.mean()) if coverage.size else 0.0
    if not bool(valid.any()) or usable <= 0.0:
        return _claim_review("empty_coverage", cell, claim, slot,
                             {"usable_coverage": usable, "source_pixels":
                              int(source_valid.sum())})
    rectification = bound.context.rectification_for(axis)
    provenance = (
        f"claim:{claim.claim_sha256}",
        f"face:{claim.face_sha256}",
        f"frame:{claim.frame}",
        f"image:{bound.context.pixels_sha256}",
        f"mask:{bound.context.usable_mask_sha256}",
        f"reading:{claim.reading_identity_sha256}",
        f"rectification:{rectification.rectification_sha256}",
        f"region:{claim.source_region_id}",
        f"support:{claim.support_sha256}",
        f"volume:{volume_sha256}",
    )
    try:
        observation = FaceObservation(
            cell=cell.coordinate, slot=slot, frame_id=claim.frame,
            normal=NORMAL_OF_SLOT[slot], patch=patch, usable_coverage=usable,
            support_frames=support_frames, valid=valid, coverage=coverage,
            provenance=provenance, weight=1.0)
    except (TypeError, ValueError) as error:
        raise _InputRefusal("invalid_input", {"field": "observation"}) from error
    recorder.tick("payload_bytes", _observation_payload_bytes(observation))
    return observation


def _binding(
    claim: pv.CellClaim,
    slot: str,
    observation: FaceObservation,
    bound: _BoundSource,
) -> FaceObservationBinding:
    rectification = bound.context.rectification_for(int(claim.reading_axis))
    return FaceObservationBinding.build(
        cell=tuple(claim.coordinate), slot=slot, frame_id=claim.frame,
        reading_axis=int(claim.reading_axis),
        world_normal_axis=int(claim.world_normal_axis),
        world_normal_sign=int(claim.world_normal_sign),
        outward_sign=int(claim.outward_sign),
        row=int(claim.row), column=int(claim.column),
        claim_sha256=claim.claim_sha256,
        reading_identity_sha256=claim.reading_identity_sha256,
        support_sha256=claim.support_sha256,
        face_sha256=claim.face_sha256,
        frame_context_sha256=bound.context.context_sha256,
        rectification_sha256=rectification.rectification_sha256,
        image_pixels_sha256=bound.context.pixels_sha256,
        usable_mask_sha256=bound.context.usable_mask_sha256,
        observation_sha256=face_observation_sha256(observation))


def _unsupported_slot_review(
    cell: pv.SupportedCellObservation, slot: str,
) -> FaceObservationReview:
    return FaceObservationReview.build(
        reason="unsupported_slot", scope="slot", coordinate=cell.coordinate,
        slot=slot, frame_ids=tuple(cell.frame_ids),
        claim_sha256s=tuple(claim.claim_sha256 for claim in cell.claims),
        support_sha256s=tuple(sorted({claim.support_sha256 for claim in cell.claims})),
        detail={})


def _conflict_review(cell: pv.SupportedCellObservation) -> FaceObservationReview:
    return FaceObservationReview.build(
        reason="conflict_cell", scope="cell", coordinate=cell.coordinate,
        slot=None, frame_ids=tuple(cell.frame_ids),
        claim_sha256s=tuple(claim.claim_sha256 for claim in cell.claims),
        support_sha256s=tuple(sorted({claim.support_sha256 for claim in cell.claims})),
        detail={"conflict_kind": cell.conflict_kind})


def _unknown_review(omitted: pv.OmittedCell) -> FaceObservationReview:
    return FaceObservationReview.build(
        reason="unknown_cell", scope="cell", coordinate=None, slot=None,
        frame_ids=(omitted.frame,), claim_sha256s=(),
        support_sha256s=(omitted.support_sha256,),
        detail={
            "frame": omitted.frame,
            "plane_index": int(omitted.plane_index),
            "reading_axis": int(omitted.reading_axis),
            "row": int(omitted.row),
            "column": int(omitted.column),
            "state": omitted.state,
            "evidence": omitted.evidence,
            "source_region_id": omitted.source_region_id,
            "reading_identity_sha256": omitted.reading_identity_sha256,
            "face_sha256": omitted.face_sha256,
        })


@dataclass(frozen=True, slots=True)
class _ObservationRecords:
    observations: tuple[FaceObservation, ...]
    bindings: tuple[FaceObservationBinding, ...]
    reviews: tuple[FaceObservationReview, ...]


def _derive_records(
    authoritative: pv.PlaneVolumeResult,
    bound: Mapping[str, _BoundSource],
    recorder: Any,
) -> _ObservationRecords:
    """Emit every observation, binding and review, ticking ``recorder`` in order."""
    observations: list[FaceObservation] = []
    bindings: list[FaceObservationBinding] = []
    reviews: list[FaceObservationReview] = []
    for cell in authoritative.cells:
        if cell.conflict_kind is not None:
            reviews.append(_conflict_review(cell))
            continue
        grouped: dict[str, list[pv.CellClaim]] = {}
        for claim in cell.claims:
            slot = slot_of_normal(int(claim.world_normal_axis), int(claim.outward_sign))
            grouped.setdefault(slot, []).append(claim)
        for slot in FACE_SLOTS:
            group = grouped.get(slot)
            if not group:
                reviews.append(_unsupported_slot_review(cell, slot))
                continue
            support_frames = tuple(sorted({claim.frame for claim in group}))
            for claim in group:
                recorder.tick("claims")
                source = bound.get(claim.support_sha256)
                if source is None:
                    raise _InputRefusal("missing_provenance", {"field": "support"})
                outcome = _rectify_claim(cell, claim, slot, support_frames, source,
                                         authoritative.volume_sha256, recorder)
                if isinstance(outcome, FaceObservation):
                    observations.append(outcome)
                    bindings.append(_binding(claim, slot, outcome, source))
                else:
                    reviews.append(outcome)
    for omitted in authoritative.omitted:
        reviews.append(_unknown_review(omitted))
    ordered_observations = tuple(sorted(observations, key=face_observation_sort_key))
    ordered_bindings = tuple(sorted(bindings, key=_binding_sort_key))
    ordered_reviews = tuple(sorted(reviews, key=_review_sort_key))
    if len(ordered_observations):
        recorder.tick("observations", len(ordered_observations))
    if len(ordered_reviews):
        recorder.tick("reviews", len(ordered_reviews))
    return _ObservationRecords(ordered_observations, ordered_bindings, ordered_reviews)


def _build_result(
    authoritative: pv.PlaneVolumeResult,
    frames: Mapping[str, FrameRectificationContext],
    records: _ObservationRecords,
    work: PlaneFaceObservationWork,
) -> PlaneFaceObservationResult:
    used_axes_by_frame: dict[str, set[int]] = {}
    for binding in authoritative.support_bindings:
        used_axes_by_frame.setdefault(binding["frame"], set()).add(
            int(binding["reading_axis"]))
    frame_bindings = []
    for frame_id in sorted(used_axes_by_frame):
        context = frames.get(frame_id)
        if context is None:
            raise _InputRefusal("missing_frame_context", {"frame": frame_id})
        frame_bindings.append(_frame_binding(context, sorted(used_axes_by_frame[frame_id])))
    return PlaneFaceObservationResult.build(
        schema_version=M5_FACE_OBSERVATION_SCHEMA,
        producer_version=M5_FACE_OBSERVATION_PRODUCER_VERSION,
        coverage_policy=M5_FACE_OBSERVATION_COVERAGE_POLICY,
        volume_sha256=authoritative.volume_sha256,
        placement_sha256=authoritative.placement_sha256,
        frame_bindings=tuple(frame_bindings),
        observations=records.observations,
        bindings=records.bindings,
        reviews=records.reviews,
        work=work)


def _apply_budget(budget: PlaneFaceObservationBudget,
                  events: Sequence[tuple]) -> PlaneFaceObservationRefusal | None:
    """Replay recorded work events through a stop-on-cap counter."""
    counter = _Counter(budget)
    try:
        for event in events:
            counter.tick(event[1], event[2])
    except _BudgetStop as stop:
        return _make_refusal("budget_exhausted", {
            "cap": stop.cap, "limit": stop.limit,
            "observed": stop.observed}, counter.snapshot())
    return None


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_plane_face_observations(
    result: pv.PlaneVolumeResult,
    *,
    universe: plane_placement.PlacementUniverse,
    supports: Sequence[v1.AlignedRegionSupport],
    placement: plane_placement.ResolvedFacePlacement,
    placement_budget: plane_placement.PlacementBudget,
    volume_budget: pv.PlaneVolumeBudget,
    frames: Sequence[FrameRectificationContext],
    budget: PlaneFaceObservationBudget,
    producer_version: str = M5_FACE_OBSERVATION_PRODUCER_VERSION,
) -> PlaneFaceObservationResult | PlaneFaceObservationRefusal:
    """Bridge one accepted V3 volume into immutable plane-homography observations.

    The accepted V3 result is rerun from the original universe, supports,
    placement and budgets and must replay byte for byte; every frame context and
    support is then bound to its original reading, region mask, usable mask,
    calibration and image digest; and only after all of that is the caller's
    bridge budget applied.  Each ``(cell, physical slot, frame claim)`` becomes
    one immutable :class:`~face_observation.FaceObservation`, unknown/conflicting
    cells and unclaimed slots become typed review records, and a cap excess
    discards everything atomically as a ``budget_exhausted`` refusal.
    """
    if not isinstance(budget, PlaneFaceObservationBudget):
        return _make_refusal("invalid_input", {"field": "budget"})
    recorder = _RecordingCounter()
    try:
        if (type(producer_version) is not str
                or producer_version != M5_FACE_OBSERVATION_PRODUCER_VERSION):
            raise _InputRefusal("invalid_input", {"field": "producer_version"})
        authoritative = _replay_volume(
            result, universe=universe, supports=supports, placement=placement,
            placement_budget=placement_budget, volume_budget=volume_budget)
        bound = _bind_contexts(frames, authoritative, universe, supports, recorder)
        records = _derive_records(authoritative, bound, recorder)
        frames_by_id = {context.frame_id: context for context in frames}
        built = _build_result(authoritative, frames_by_id, records,
                              _make_work(recorder.counts))
    except _InputRefusal as refusal:
        return _make_refusal(refusal.reason, refusal.detail)
    except _BudgetStop as stop:  # defensive: the recorder never stops
        return _make_refusal("invalid_input", {"field": "counter", "cap": stop.cap})
    except (AttributeError, TypeError, ValueError, IndexError, KeyError,
            OverflowError, ZeroDivisionError):
        return _make_refusal("invalid_input", {"field": "request"})

    refusal = _apply_budget(budget, recorder.events)
    if refusal is not None:
        return refusal
    return built
