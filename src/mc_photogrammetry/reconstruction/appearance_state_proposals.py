"""Bounded M5 appearance/state proposal adapter.

This module is the narrow seam that turns accepted, immutable
:class:`~mc_photogrammetry.reconstruction.face_observation.FaceObservation`
tuples into typed per-cell material proposals. It consumes nothing else: an
explicit :class:`~mc_photogrammetry.blocks.appearance.PaletteColor` tuple and a
handful of explicit thresholds and budgets. Every observation is delegated,
unchanged, to the existing neutral scorer
:class:`~mc_photogrammetry.reconstruction.appearance_scoring.AppearanceScorer`,
whose :class:`~mc_photogrammetry.reconstruction.appearance_scoring.FaceScore`
is the only evidence this adapter reasons over.

What a proposal is, and is not
-------------------------------

A :class:`FaceStateProposal` is *material evidence about one cell*, never an
occupancy verdict. It carries a proposed :class:`BlockState` only when the
aggregate evidence across the cell's usable observations is both cheap enough
and separated enough from its runner-up; otherwise it is a typed review
abstention with one of :data:`ABSTENTION_REASONS`. The adapter never infers
occupancy, hidden blocks, geometry, camera pose, scale, bounds or manually
supplied metadata, and it never fuses or exports a volume. ``BlockState``
appears here only as the proposed value.

Evidence discipline
-------------------

* Inputs are validated in full *before* any budget is consulted, so a malformed
  observation or palette is a typed error rather than a silently dropped cell.
  A retained array must be the accepted bridge's exact artifact: ``float32``
  patch, boolean ``valid``, ``float64`` coverage, read-only, and backed by
  immutable ``bytes`` (non-reopenable). Nothing is copied or coerced to pass.
* Observations are grouped by cell; a cell's proposal consumes only that cell's
  observations, and observations of one cell are never aggregated into another.
* Aggregation is keyed by the canonical :class:`BlockState` identity, so two
  spellings of the same state are one candidate and palette order cannot change
  the winner.
* Support frame ids, observation digests, an aggregate evidence digest, the
  exact provenance strings and every per-candidate aggregate cost are retained.
* The observation digest binds each retained array's dtype, shape and exact
  C-order bytes, so a proposal cannot be replayed against reshaped, retyped or
  otherwise different evidence under the same value.
* Every budget refusal is atomic: a refused call emits no partial proposal.

Aggregate rule
--------------

For each usable observation the scorer returns relative per-candidate costs and
the absolute cost of its own best candidate. A candidate's absolute cost under
one observation is ``best_cost + relative_cost``; its aggregate cost is the
equal-weight mean of those absolute costs over the cell's usable observations.
The aggregate best cost is therefore on the same scale as ``max_cost`` no matter
how many observations a cell has, and the margin is the aggregate runner-up cost
minus the aggregate best cost.

Precedence is explicit and deterministic: ``no_observation`` before anything
else; then ``high_cost`` when the aggregate best cost is not finite or exceeds
``max_cost``; then ``weak_margin`` when the margin is below ``min_margin``; then
``conflicting_faces`` when two usable observations of the *same slot* each
nominate a different confident state; otherwise the aggregate winner is
proposed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping

import numpy as np

from mc_photogrammetry.blocks.appearance import PaletteColor
from mc_photogrammetry.blocks.model_atlas import FACE_SLOTS
from mc_photogrammetry.domain.models import BlockCoordinate, BlockState

from .appearance_scoring import AppearanceScorer, FaceScore
from .face_observation import NORMAL_OF_SLOT, FaceObservation

PROPOSAL_SCHEMA = "m5-appearance-state-proposal-1"
RESULT_SCHEMA = "m5-appearance-state-proposal-result-1"
EVIDENCE_SCHEMA = "m5-appearance-state-evidence-1"
OBSERVATION_SCHEMA = "m5-face-observation-1"
PRODUCER_VERSION = "m5-appearance-state-proposal-adapter-1"

PROPOSED = "proposed"
"""Status of a proposal that carries an accepted state."""

REVIEW = "review"
"""Status of a typed per-cell abstention that carries no state."""

REFUSED = "budget"
"""Status of an atomic result-level budget refusal."""

NO_OBSERVATION = "no_observation"
EMPTY_PALETTE = "empty_palette"
WEAK_MARGIN = "weak_margin"
HIGH_COST = "high_cost"
CONFLICTING_FACES = "conflicting_faces"
BUDGET = "budget"

ABSTENTION_REASONS: tuple[str, ...] = (
    NO_OBSERVATION,
    EMPTY_PALETTE,
    WEAK_MARGIN,
    HIGH_COST,
    CONFLICTING_FACES,
    BUDGET,
)
"""Every typed review/abstention reason this adapter can emit."""

MAX_OBSERVATIONS = "max_observations"
MAX_CANDIDATE_EVALUATIONS = "max_candidate_evaluations"
MAX_PAYLOAD_BYTES = "max_payload_bytes"

BUDGETS: tuple[str, ...] = (MAX_OBSERVATIONS, MAX_CANDIDATE_EVALUATIONS, MAX_PAYLOAD_BYTES)
"""Every explicit budget whose exhaustion refuses the call atomically."""


class AppearanceStateProposalError(ValueError):
    """A malformed observation, palette, request or scorer result.

    Raised before any result can exist, so it is never a partial outcome.
    """


@dataclass(frozen=True, slots=True)
class FaceStateProposal:
    """One cell's accepted material proposal or typed review abstention."""

    cell: BlockCoordinate
    """The single cell all retained evidence belongs to."""

    status: str
    """``"proposed"`` or ``"review"``."""

    reason: str | None
    """The typed abstention reason, or ``None`` for a proposal."""

    state: BlockState | None
    """The proposed exact state, or ``None`` for a review abstention."""

    best_cost: float
    """Aggregate absolute cost of the winning candidate (``inf`` when unobserved)."""

    runner_up_state: BlockState | None
    """Aggregate runner-up, or ``None`` when the palette has one candidate."""

    runner_up_cost: float
    """Aggregate absolute cost of the runner-up (``inf`` when there is none)."""

    margin: float
    """``runner_up_cost - best_cost`` (``inf`` with a single candidate)."""

    candidate_costs: tuple[tuple[BlockState, float], ...]
    """Aggregate absolute cost per canonical state, cheapest first."""

    support_frame_ids: tuple[str, ...]
    """Sorted union of every retained observation's support frames."""

    observation_digests: tuple[str, ...]
    """Sorted content digests of every retained observation, usable or not."""

    provenance: tuple[str, ...]
    """Sorted union of the observations' exact provenance references."""

    evidence_digest: str
    """Digest binding the cell, palette and the retained observation entries."""

    usable_observation_count: int
    """How many of this cell's observations could vote."""

    observation_count: int
    """How many observations this proposal retained, usable or not."""

    slots: tuple[str, ...]
    """Sorted world-face slots observed on this cell (all observations)."""

    max_cost: float
    """The explicit best-cost ceiling this proposal was decided under."""

    min_margin: float
    """The explicit margin floor this proposal was decided under."""

    proposal_digest: str
    """Deterministic digest of every field above."""

    def __post_init__(self) -> None:
        if self.status not in (PROPOSED, REVIEW):
            raise AppearanceStateProposalError(f"unknown proposal status {self.status!r}")
        if self.status == PROPOSED:
            if not isinstance(self.state, BlockState):
                raise AppearanceStateProposalError("a proposed state must be a BlockState")
            if self.reason is not None:
                raise AppearanceStateProposalError("a proposal cannot carry a reason")
        else:
            if self.state is not None:
                raise AppearanceStateProposalError("a review abstention cannot carry a state")
            if self.reason not in ABSTENTION_REASONS:
                raise AppearanceStateProposalError(f"unknown abstention reason {self.reason!r}")
        if not _is_cell(self.cell):
            raise AppearanceStateProposalError("proposal cell must be a 3-tuple of integers")
        if self.runner_up_state is not None and not isinstance(self.runner_up_state, BlockState):
            raise AppearanceStateProposalError("runner_up_state must be a BlockState or None")
        if self.best_cost < 0.0 or math.isnan(self.best_cost):
            raise AppearanceStateProposalError("best_cost must be non-negative and not NaN")
        if self.runner_up_cost < 0.0 or math.isnan(self.runner_up_cost):
            raise AppearanceStateProposalError("runner_up_cost must be non-negative and not NaN")
        if math.isnan(self.margin):
            raise AppearanceStateProposalError("margin must not be NaN")
        for state, cost in self.candidate_costs:
            if not isinstance(state, BlockState):
                raise AppearanceStateProposalError("candidate costs must be keyed by BlockState")
            if not math.isfinite(cost) or cost < 0.0:
                raise AppearanceStateProposalError(
                    "candidate costs must be finite and non-negative"
                )
        for digest in (self.evidence_digest, self.proposal_digest):
            if not _is_hex64(digest):
                raise AppearanceStateProposalError("proposal digests must be SHA-256 digests")
        for digest in self.observation_digests:
            if not _is_hex64(digest):
                raise AppearanceStateProposalError("observation digests must be SHA-256 digests")
        expected = _sha256(_proposal_payload(self))
        if self.proposal_digest != expected:
            raise AppearanceStateProposalError("proposal digest mismatch")

    @property
    def proposed(self) -> bool:
        return self.status == PROPOSED

    def cost_for(self, state: BlockState) -> float | None:
        for candidate, cost in self.candidate_costs:
            if candidate == state:
                return cost
        return None

    def ranked(self) -> tuple[tuple[BlockState, float], ...]:
        """Aggregate candidates cheapest first, canonical-state ties included."""
        return self.candidate_costs


@dataclass(frozen=True, slots=True)
class FaceStateProposalResult:
    """The immutable, atomic outcome of one bounded proposal call."""

    status: str
    """``"proposed"`` when any cell was accepted, ``"review"`` when none was,
    ``"budget"`` when the call was refused."""

    reason: str | None
    """``"no_observation"``/``"empty_palette"``/``"budget"`` at result level, else ``None``."""

    budget: str | None
    """Which explicit budget refused the call, or ``None``."""

    proposals: tuple[FaceStateProposal, ...]
    """Per-cell proposals in canonical cell order; empty on any refusal."""

    observation_count: int
    """Total observations validated by this call."""

    palette_size: int
    """Total palette entries validated by this call."""

    candidate_evaluations: int
    """Observation-by-candidate evaluations actually performed before emitting."""

    payload_bytes: int
    """Canonical encoded size of the emitted result payload."""

    payload_sha256: str
    """Digest of the canonical result payload."""

    result_digest: str
    """Digest binding the payload, its digest and the explicit byte budget."""

    max_cost: float
    """The explicit best-cost ceiling this call was made under."""

    min_margin: float
    """The explicit margin floor this call was made under."""

    max_observations: int
    """The explicit observation budget this call was made under."""

    max_candidate_evaluations: int
    """The explicit candidate-evaluation budget this call was made under."""

    max_payload_bytes: int
    """The explicit payload-byte budget this call was made under."""

    def __post_init__(self) -> None:
        if self.status not in (PROPOSED, REVIEW, REFUSED):
            raise AppearanceStateProposalError(f"unknown result status {self.status!r}")
        if self.status == REFUSED:
            if self.budget not in BUDGETS or self.reason != BUDGET or self.proposals:
                raise AppearanceStateProposalError("a budget refusal must be atomic and typed")
        elif self.budget is not None:
            raise AppearanceStateProposalError("only a budget refusal may name a budget")
        if self.reason is not None and self.reason not in ABSTENTION_REASONS:
            raise AppearanceStateProposalError(f"unknown result reason {self.reason!r}")
        for proposal in self.proposals:
            if not isinstance(proposal, FaceStateProposal):
                raise AppearanceStateProposalError("proposals must be FaceStateProposal records")
        if self.observation_count < 0 or self.palette_size < 0 or self.candidate_evaluations < 0:
            raise AppearanceStateProposalError("result counters must be non-negative")
        if self.payload_bytes < 0:
            raise AppearanceStateProposalError("payload_bytes must be non-negative")
        for digest in (self.payload_sha256, self.result_digest):
            if not _is_hex64(digest):
                raise AppearanceStateProposalError("result digests must be SHA-256 digests")
        encoded = _canonical_json(_result_payload(self))
        if len(encoded.encode("utf-8")) != self.payload_bytes:
            raise AppearanceStateProposalError("result payload byte count mismatch")
        if _sha256(encoded) != self.payload_sha256:
            raise AppearanceStateProposalError("result payload digest mismatch")
        if _sha256(_result_digest_payload(self)) != self.result_digest:
            raise AppearanceStateProposalError("result digest mismatch")


# --- canonical primitives -----------------------------------------------------------------


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_hex64(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _is_cell(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 3
        and all(isinstance(entry, int) and not isinstance(entry, bool) for entry in value)
    )


def _is_real(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _state_key(state: BlockState) -> tuple[str, tuple[tuple[str, str], ...]]:
    """The canonical identity ``BlockState`` equality and hashing already use."""
    identity = state._identity
    if identity is None:  # pragma: no cover - always set by BlockState.__post_init__
        return (state.name, tuple(sorted(state.properties)))
    return identity


def _state_payload(state: BlockState) -> dict[str, Any]:
    name, properties = _state_key(state)
    return {"name": name, "properties": [list(entry) for entry in properties]}


def _palette_payload(item: PaletteColor) -> dict[str, Any]:
    """Only the fields that can change a score under this adapter's settings."""
    return {
        "state": _state_payload(item.state),
        "rgb": [int(channel) for channel in item.rgb],
        "face_rgbs": [[int(channel) for channel in face] for face in item.face_rgbs],
        "face_map": sorted(
            ([str(slot), [int(channel) for channel in face]] for slot, face in item.face_map),
            key=lambda entry: entry[0],
        ),
        "family": str(item.family),
        "descriptors": [[float(value) for value in row] for row in item.descriptors],
        "full_cube": bool(item.full_cube),
        "shape_boxes": [
            [[float(value) for value in low], [float(value) for value in high]]
            for low, high in item.shape_boxes
        ],
        "covered_slots": sorted(str(slot) for slot in item.covered_slots),
    }


def _palette_digest(palette: tuple[PaletteColor, ...]) -> str:
    items = sorted((_palette_payload(item) for item in palette), key=_canonical_json)
    return _sha256({"schema": "m5-palette-1", "entries": items})


def _retained_array_payload(array: np.ndarray | None) -> dict[str, Any] | None:
    """Bind one retained array's exact dtype, shape and C-order bytes.

    Hashing the bytes alone would let two distinct arrays collide: a
    ``(2, 1, 3)`` patch and a ``(1, 2, 3)`` patch of the same six values
    serialize identically, and an ``int32`` and a ``float32`` array of the same
    shape share a byte stream. The dtype string and shape are therefore part of
    the digest alongside the bytes.
    """
    if array is None:
        return None
    return {
        "dtype": array.dtype.str,
        "shape": [int(dimension) for dimension in array.shape],
        "sha256": _sha256_bytes(array.tobytes(order="C")),
    }


def _observation_digest(observation: FaceObservation) -> str:
    return _sha256(
        {
            "schema": OBSERVATION_SCHEMA,
            "cell": [int(entry) for entry in observation.cell],
            "slot": observation.slot,
            "frame_id": observation.frame_id,
            "normal": [float(value) for value in observation.normal],
            "usable_coverage": float(observation.usable_coverage),
            "weight": float(observation.weight),
            "support_frames": list(observation.support_frames),
            "provenance": list(observation.provenance),
            "patch": _retained_array_payload(observation.patch),
            "valid": _retained_array_payload(observation.valid),
            "coverage": _retained_array_payload(observation.coverage),
        }
    )


def _observation_order_key(observation: FaceObservation) -> tuple[Any, ...]:
    return (
        observation.cell,
        observation.slot,
        observation.frame_id,
        _observation_digest(observation),
    )


def _evidence_digest(
    cell: BlockCoordinate,
    palette_digest: str,
    entries: list[dict[str, Any]],
) -> str:
    return _sha256(
        {
            "schema": EVIDENCE_SCHEMA,
            "cell": [int(entry) for entry in cell],
            "palette_sha256": palette_digest,
            "observations": entries,
        }
    )


def _proposal_payload(source: Any) -> dict[str, Any]:
    def read(name: str) -> Any:
        if isinstance(source, Mapping):
            return source[name]
        return getattr(source, name)

    state = read("state")
    runner_up = read("runner_up_state")
    return {
        "schema": PROPOSAL_SCHEMA,
        "producer": PRODUCER_VERSION,
        "cell": [int(entry) for entry in read("cell")],
        "status": read("status"),
        "reason": read("reason"),
        "state": None if state is None else _state_payload(state),
        "best_cost": float(read("best_cost")),
        "runner_up_state": None if runner_up is None else _state_payload(runner_up),
        "runner_up_cost": float(read("runner_up_cost")),
        "margin": float(read("margin")),
        "candidate_costs": [
            [_state_payload(candidate), float(cost)] for candidate, cost in read("candidate_costs")
        ],
        "support_frame_ids": list(read("support_frame_ids")),
        "observation_digests": list(read("observation_digests")),
        "provenance": list(read("provenance")),
        "evidence_sha256": read("evidence_digest"),
        "usable_observation_count": int(read("usable_observation_count")),
        "observation_count": int(read("observation_count")),
        "slots": list(read("slots")),
        "max_cost": float(read("max_cost")),
        "min_margin": float(read("min_margin")),
    }


def _result_payload_values(
    *,
    status: str,
    reason: str | None,
    budget: str | None,
    proposals: tuple[FaceStateProposal, ...],
    observation_count: int,
    palette_size: int,
    candidate_evaluations: int,
    max_cost: float,
    min_margin: float,
    max_observations: int,
    max_candidate_evaluations: int,
) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "producer": PRODUCER_VERSION,
        "status": status,
        "reason": reason,
        "budget": budget,
        "proposals": [_proposal_payload(proposal) for proposal in proposals],
        "observation_count": int(observation_count),
        "palette_size": int(palette_size),
        "candidate_evaluations": int(candidate_evaluations),
        "max_cost": float(max_cost),
        "min_margin": float(min_margin),
        "max_observations": int(max_observations),
        "max_candidate_evaluations": int(max_candidate_evaluations),
    }


def _result_payload(result: FaceStateProposalResult) -> dict[str, Any]:
    return _result_payload_values(
        status=result.status,
        reason=result.reason,
        budget=result.budget,
        proposals=result.proposals,
        observation_count=result.observation_count,
        palette_size=result.palette_size,
        candidate_evaluations=result.candidate_evaluations,
        max_cost=result.max_cost,
        min_margin=result.min_margin,
        max_observations=result.max_observations,
        max_candidate_evaluations=result.max_candidate_evaluations,
    )


def _result_digest_payload(result: FaceStateProposalResult) -> dict[str, Any]:
    return {
        **_result_payload(result),
        "payload_sha256": result.payload_sha256,
        "max_payload_bytes": result.max_payload_bytes,
    }


# --- validation ---------------------------------------------------------------------------


def _validate_request(
    max_cost: Any,
    min_margin: Any,
    max_observations: Any,
    max_candidate_evaluations: Any,
    max_payload_bytes: Any,
) -> None:
    for name, value in (("max_cost", max_cost), ("min_margin", min_margin)):
        if not _is_real(value) or not math.isfinite(float(value)) or float(value) < 0.0:
            raise AppearanceStateProposalError(f"{name} must be a finite non-negative number")
    for name, value in (
        (MAX_OBSERVATIONS, max_observations),
        (MAX_CANDIDATE_EVALUATIONS, max_candidate_evaluations),
        (MAX_PAYLOAD_BYTES, max_payload_bytes),
    ):
        if not isinstance(value, Integral) or isinstance(value, bool) or int(value) < 0:
            raise AppearanceStateProposalError(f"{name} must be a non-negative integer")


def _validate_observations(observations: Any) -> None:
    if not isinstance(observations, tuple):
        raise AppearanceStateProposalError("observations must be an immutable tuple")
    for index, observation in enumerate(observations):
        _validate_observation(index, observation)


def _has_immutable_bytes_base(array: np.ndarray) -> bool:
    """True when ``array``'s ``base`` chain terminates at immutable ``bytes``.

    A numpy-owned array (``owndata is True``) or a view into another numpy
    array can always be reopened with ``setflags(write=True)``; only a buffer
    whose ultimate base is ``bytes`` is genuinely non-reopenable. This mirrors
    the accepted face-observation bridge, whose retained arrays are
    ``bytes``-backed read-only views.
    """
    base: Any = array
    while base is not None:
        if isinstance(base, bytes):
            return True
        if not isinstance(base, np.ndarray):
            return False
        base = base.base
    return False


def _require_immutable_retained(label: str, name: str, array: np.ndarray) -> None:
    """Enforce the accepted retained-array invariant without copying or coercion.

    A later ``setflags(write=True)`` on numpy-owned memory could change a score
    after it was computed, so the adapter checks the exact array it was handed:
    the writeable flag must be clear *and* the base chain must terminate at
    ``bytes``. Nothing is copied or normalised to satisfy the check.
    """
    if array.flags.writeable:
        raise AppearanceStateProposalError(
            f"{label} {name} must be a read-only retained array (writeable=False)"
        )
    if not _has_immutable_bytes_base(array):
        raise AppearanceStateProposalError(
            f"{label} {name} must be backed by immutable bytes, not reopenable numpy-owned memory"
        )


def _validate_observation(index: int, observation: Any) -> None:
    if not isinstance(observation, FaceObservation):
        raise AppearanceStateProposalError(
            f"observation {index} must be a FaceObservation, got {type(observation).__name__}"
        )
    label = f"observation {index}"
    try:
        cell = observation.cell
        slot = observation.slot
        frame_id = observation.frame_id
        normal = observation.normal
        patch = observation.patch
        valid = observation.valid
        coverage = observation.coverage
        usable = observation.usable_coverage
        support = observation.support_frames
        provenance = observation.provenance
        weight = observation.weight
    except AttributeError as error:  # pragma: no cover - tampered/partial record
        raise AppearanceStateProposalError(f"{label} is malformed: {error}") from error

    if not _is_cell(cell):
        raise AppearanceStateProposalError(f"{label} cell must be a 3-tuple of integers")
    if slot not in FACE_SLOTS:
        raise AppearanceStateProposalError(f"{label} slot must be one of {FACE_SLOTS}")
    if not isinstance(frame_id, str) or not frame_id:
        raise AppearanceStateProposalError(f"{label} frame_id must be a non-empty string")
    if not isinstance(normal, tuple) or len(normal) != 3:
        raise AppearanceStateProposalError(f"{label} normal must be a 3-tuple")
    normal_values = np.asarray(normal, dtype=np.float64)
    if not np.isfinite(normal_values).all():
        raise AppearanceStateProposalError(f"{label} normal must be finite")
    if abs(float(np.linalg.norm(normal_values)) - 1.0) > 1e-6:
        raise AppearanceStateProposalError(f"{label} normal must be a unit vector")
    if not np.allclose(
        normal_values, np.asarray(NORMAL_OF_SLOT[slot], dtype=np.float64), atol=1e-6
    ):
        raise AppearanceStateProposalError(f"{label} normal is inconsistent with slot {slot!r}")

    if (
        not isinstance(patch, np.ndarray)
        or patch.ndim != 3
        or patch.shape[2] != 3
        or patch.shape[0] < 1
        or patch.shape[1] < 1
    ):
        raise AppearanceStateProposalError(
            f"{label} patch must be a non-empty (height, width, 3) array"
        )
    if patch.dtype != np.float32:
        raise AppearanceStateProposalError(f"{label} patch must be a float32 array")
    _require_immutable_retained(label, "patch", patch)
    if not np.isfinite(patch).all():
        raise AppearanceStateProposalError(f"{label} patch must be finite")
    if valid is not None:
        if not isinstance(valid, np.ndarray) or valid.dtype != np.bool_:
            raise AppearanceStateProposalError(f"{label} valid must be a boolean array")
        if valid.shape != patch.shape[:2]:
            raise AppearanceStateProposalError(
                f"{label} valid must match the patch height and width"
            )
        _require_immutable_retained(label, "valid", valid)
    if coverage is not None:
        if not isinstance(coverage, np.ndarray) or coverage.shape != patch.shape[:2]:
            raise AppearanceStateProposalError(
                f"{label} coverage must match the patch height and width"
            )
        if coverage.dtype != np.float64:
            raise AppearanceStateProposalError(f"{label} coverage must be a float64 array")
        _require_immutable_retained(label, "coverage", coverage)
        if not np.isfinite(coverage).all():
            raise AppearanceStateProposalError(f"{label} coverage must be finite")
        if float(coverage.min()) < -1e-9 or float(coverage.max()) > 1.0 + 1e-9:
            raise AppearanceStateProposalError(f"{label} coverage must lie in [0, 1]")

    if not _is_real(usable) or not math.isfinite(float(usable)):
        raise AppearanceStateProposalError(f"{label} usable_coverage must be a finite number")
    if float(usable) < -1e-9 or float(usable) > 1.0 + 1e-9:
        raise AppearanceStateProposalError(f"{label} usable_coverage must lie in [0, 1]")

    if (
        not isinstance(support, tuple)
        or not support
        or len(set(support)) != len(support)
        or any(not isinstance(name, str) or not name for name in support)
    ):
        raise AppearanceStateProposalError(
            f"{label} support_frames must be a non-empty tuple of unique frame ids"
        )
    if frame_id not in support:
        raise AppearanceStateProposalError(f"{label} support_frames must include frame_id")
    if (
        not isinstance(provenance, tuple)
        or len(set(provenance)) != len(provenance)
        or any(not isinstance(name, str) or not name for name in provenance)
    ):
        raise AppearanceStateProposalError(
            f"{label} provenance must be a tuple of unique references"
        )
    if not _is_real(weight) or not math.isfinite(float(weight)) or float(weight) < 0.0:
        raise AppearanceStateProposalError(f"{label} weight must be finite and non-negative")


def _validate_palette(palette: Any) -> tuple[PaletteColor, ...]:
    if not isinstance(palette, tuple):
        raise AppearanceStateProposalError("palette must be an immutable tuple of PaletteColor")
    for index, item in enumerate(palette):
        if not isinstance(item, PaletteColor):
            raise AppearanceStateProposalError(
                f"palette entry {index} must be a PaletteColor, got {type(item).__name__}"
            )
        if not isinstance(item.state, BlockState):
            raise AppearanceStateProposalError(f"palette entry {index} must carry a BlockState")
    return palette


# --- scoring and aggregation --------------------------------------------------------------


def _score_observations(
    observations: tuple[FaceObservation, ...],
    palette: tuple[PaletteColor, ...],
) -> list[FaceScore | None]:
    """Score every usable observation once, in deterministic observation order."""
    if not any(observation.is_usable for observation in observations):
        return [None] * len(observations)
    try:
        scorer = AppearanceScorer(palette)
    except AppearanceStateProposalError:
        raise
    except (TypeError, ValueError, KeyError, IndexError) as error:
        raise AppearanceStateProposalError(
            f"appearance scorer rejected the palette: {error}"
        ) from error
    scores: list[FaceScore | None] = []
    for observation in observations:
        if not observation.is_usable:
            scores.append(None)
            continue
        try:
            raw = scorer.score(observation)
        except AppearanceStateProposalError:
            raise
        except (TypeError, ValueError, KeyError, IndexError) as error:
            raise AppearanceStateProposalError(f"appearance scorer failed: {error}") from error
        scores.append(_validate_score(raw, palette))
    return scores


def _validate_score(raw: Any, palette: tuple[PaletteColor, ...]) -> FaceScore:
    """Bind the scorer's output exactly, or stop with a typed error."""
    if not isinstance(raw, FaceScore):
        raise AppearanceStateProposalError(
            f"appearance scorer must return a FaceScore, got {type(raw).__name__}"
        )
    if raw.abstained:
        raise AppearanceStateProposalError(
            "appearance scorer abstained for an observation it reported usable"
        )
    if not isinstance(raw.costs, tuple) or len(raw.costs) != len(palette):
        raise AppearanceStateProposalError("appearance scorer must score every palette entry once")
    expected = {_state_key(item.state) for item in palette}
    seen: set[Any] = set()
    for entry in raw.costs:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise AppearanceStateProposalError("face score costs must be (state, cost) tuples")
        state, cost = entry
        if not isinstance(state, BlockState):
            raise AppearanceStateProposalError("face score costs must be keyed by BlockState")
        if not _is_real(cost) or not math.isfinite(float(cost)) or float(cost) < 0.0:
            raise AppearanceStateProposalError("face score costs must be finite and non-negative")
        seen.add(_state_key(state))
    if seen != expected:
        raise AppearanceStateProposalError("face score states do not match the palette")
    best = raw.best_cost
    if not _is_real(best) or math.isnan(float(best)) or float(best) < 0.0:
        raise AppearanceStateProposalError("face score best_cost must be non-negative and not NaN")
    return raw


def _merged_costs(score: FaceScore) -> dict[Any, tuple[BlockState, float]]:
    """One canonical entry per state, the cheapest when the palette repeats one."""
    merged: dict[Any, tuple[BlockState, float]] = {}
    for state, cost in score.costs:
        key = _state_key(state)
        value = float(cost)
        current = merged.get(key)
        if current is None or value < current[1]:
            merged[key] = (state, value)
    return merged


def _ranked_merged(
    score: FaceScore,
) -> list[tuple[BlockState, float]]:
    return sorted(
        _merged_costs(score).values(),
        key=lambda entry: (entry[1], _state_key(entry[0])),
    )


def _aggregate(
    usable: list[tuple[FaceObservation, FaceScore]],
    palette: tuple[PaletteColor, ...],
) -> list[tuple[BlockState, float]]:
    """Equal-weight mean absolute cost per canonical state, cheapest first."""
    totals: dict[Any, list[Any]] = {}
    for item in palette:
        totals.setdefault(_state_key(item.state), [item.state, 0.0])
    for _observation, score in usable:
        merged = _merged_costs(score)
        for key, entry in totals.items():
            state, cost = merged[key]
            entry[1] += float(score.best_cost) + cost
    count = float(len(usable))
    return sorted(
        ((entry[0], entry[1] / count) for entry in totals.values()),
        key=lambda entry: (entry[1], _state_key(entry[0])),
    )


def _conflicting(
    scored: list[tuple[FaceObservation, FaceScore]],
    *,
    max_cost: float,
    min_margin: float,
) -> bool:
    """Whether two usable observations of one slot confidently disagree."""
    by_slot: dict[str, list[FaceScore]] = {}
    for observation, score in scored:
        by_slot.setdefault(observation.slot, []).append(score)
    for group in by_slot.values():
        if len(group) < 2:
            continue
        winners: set[Any] = set()
        for score in group:
            ranked = _ranked_merged(score)
            best_state, best_cost = ranked[0]
            if not math.isfinite(best_cost) or best_cost > max_cost:
                continue
            margin = ranked[1][1] - best_cost if len(ranked) > 1 else math.inf
            if margin < min_margin:
                continue
            winners.add(_state_key(best_state))
        if len(winners) > 1:
            return True
    return False


# --- proposal and result construction ------------------------------------------------------


def _make_proposal(**fields: Any) -> FaceStateProposal:
    digest = _sha256(_proposal_payload(fields))
    return FaceStateProposal(**fields, proposal_digest=digest)


def _propose_cell(
    cell: BlockCoordinate,
    entries: list[tuple[FaceObservation, FaceScore | None]],
    palette: tuple[PaletteColor, ...],
    palette_digest: str,
    *,
    max_cost: float,
    min_margin: float,
) -> FaceStateProposal:
    observations = [observation for observation, _score in entries]
    scored = [(observation, score) for observation, score in entries if score is not None]
    observation_digests = tuple(sorted(_observation_digest(o) for o in observations))
    support = tuple(sorted({frame for o in observations for frame in o.support_frames}))
    provenance = tuple(sorted({reference for o in observations for reference in o.provenance}))
    slots = tuple(sorted({o.slot for o in observations}))
    evidence = _evidence_digest(
        cell,
        palette_digest,
        [
            {
                "slot": o.slot,
                "frame_id": o.frame_id,
                "observation_sha256": _observation_digest(o),
                "usable_coverage": float(o.usable_coverage),
                "weight": float(o.weight),
                "usable": bool(o.is_usable),
            }
            for o in observations
        ],
    )
    common: dict[str, Any] = {
        "cell": cell,
        "support_frame_ids": support,
        "observation_digests": observation_digests,
        "provenance": provenance,
        "evidence_digest": evidence,
        "usable_observation_count": len(scored),
        "observation_count": len(observations),
        "slots": slots,
        "max_cost": float(max_cost),
        "min_margin": float(min_margin),
    }
    if not scored:
        return _make_proposal(
            status=REVIEW,
            reason=NO_OBSERVATION,
            state=None,
            best_cost=math.inf,
            runner_up_state=None,
            runner_up_cost=math.inf,
            margin=0.0,
            candidate_costs=(),
            **common,
        )

    aggregate = _aggregate(scored, palette)
    best_state, best_cost = aggregate[0]
    runner_up_state, runner_up_cost = (
        (aggregate[1][0], aggregate[1][1]) if len(aggregate) > 1 else (None, math.inf)
    )
    margin = runner_up_cost - best_cost if len(aggregate) > 1 else math.inf
    candidate_costs = tuple(aggregate)

    if not math.isfinite(best_cost) or best_cost > max_cost:
        return _make_proposal(
            status=REVIEW,
            reason=HIGH_COST,
            state=None,
            best_cost=best_cost,
            runner_up_state=runner_up_state,
            runner_up_cost=runner_up_cost,
            margin=margin,
            candidate_costs=candidate_costs,
            **common,
        )
    if margin < min_margin:
        return _make_proposal(
            status=REVIEW,
            reason=WEAK_MARGIN,
            state=None,
            best_cost=best_cost,
            runner_up_state=runner_up_state,
            runner_up_cost=runner_up_cost,
            margin=margin,
            candidate_costs=candidate_costs,
            **common,
        )
    if _conflicting(scored, max_cost=max_cost, min_margin=min_margin):
        return _make_proposal(
            status=REVIEW,
            reason=CONFLICTING_FACES,
            state=None,
            best_cost=best_cost,
            runner_up_state=runner_up_state,
            runner_up_cost=runner_up_cost,
            margin=margin,
            candidate_costs=candidate_costs,
            **common,
        )
    return _make_proposal(
        status=PROPOSED,
        reason=None,
        state=best_state,
        best_cost=best_cost,
        runner_up_state=runner_up_state,
        runner_up_cost=runner_up_cost,
        margin=margin,
        candidate_costs=candidate_costs,
        **common,
    )


def _result(
    *,
    status: str,
    reason: str | None,
    budget: str | None,
    proposals: tuple[FaceStateProposal, ...],
    observation_count: int,
    palette_size: int,
    candidate_evaluations: int,
    max_cost: float,
    min_margin: float,
    max_observations: int,
    max_candidate_evaluations: int,
    max_payload_bytes: int,
) -> FaceStateProposalResult:
    payload = _result_payload_values(
        status=status,
        reason=reason,
        budget=budget,
        proposals=proposals,
        observation_count=observation_count,
        palette_size=palette_size,
        candidate_evaluations=candidate_evaluations,
        max_cost=max_cost,
        min_margin=min_margin,
        max_observations=max_observations,
        max_candidate_evaluations=max_candidate_evaluations,
    )
    encoded = _canonical_json(payload)
    payload_sha256 = _sha256(encoded)
    result_digest = _sha256({**payload, "payload_sha256": payload_sha256,
                             "max_payload_bytes": int(max_payload_bytes)})
    result = FaceStateProposalResult(
        status=status,
        reason=reason,
        budget=budget,
        proposals=proposals,
        observation_count=observation_count,
        palette_size=palette_size,
        candidate_evaluations=candidate_evaluations,
        payload_bytes=len(encoded.encode("utf-8")),
        payload_sha256=payload_sha256,
        result_digest=result_digest,
        max_cost=float(max_cost),
        min_margin=float(min_margin),
        max_observations=int(max_observations),
        max_candidate_evaluations=int(max_candidate_evaluations),
        max_payload_bytes=int(max_payload_bytes),
    )
    return result


def propose_face_states(
    observations: tuple[FaceObservation, ...],
    palette: tuple[PaletteColor, ...],
    *,
    max_cost: float,
    min_margin: float,
    max_observations: int,
    max_candidate_evaluations: int,
    max_payload_bytes: int,
) -> FaceStateProposalResult:
    """Propose at most one exact state per observed cell, or abstain atomically.

    ``observations`` and ``palette`` are validated in full before any budget is
    consulted. ``max_cost`` and ``min_margin`` are the explicit acceptance
    thresholds; ``max_observations``, ``max_candidate_evaluations`` and
    ``max_payload_bytes`` are explicit budgets. Exceeding any budget refuses the
    whole call with ``status="budget"`` and no partial proposals.
    """
    _validate_request(
        max_cost, min_margin, max_observations, max_candidate_evaluations, max_payload_bytes
    )
    _validate_observations(observations)
    palette_items = _validate_palette(palette)

    observation_count = len(observations)
    palette_size = len(palette_items)

    if observation_count == 0:
        return _result(
            status=REVIEW,
            reason=NO_OBSERVATION,
            budget=None,
            proposals=(),
            observation_count=0,
            palette_size=palette_size,
            candidate_evaluations=0,
            max_cost=float(max_cost),
            min_margin=float(min_margin),
            max_observations=max_observations,
            max_candidate_evaluations=max_candidate_evaluations,
            max_payload_bytes=max_payload_bytes,
        )
    if palette_size == 0:
        return _result(
            status=REVIEW,
            reason=EMPTY_PALETTE,
            budget=None,
            proposals=(),
            observation_count=observation_count,
            palette_size=0,
            candidate_evaluations=0,
            max_cost=float(max_cost),
            min_margin=float(min_margin),
            max_observations=max_observations,
            max_candidate_evaluations=max_candidate_evaluations,
            max_payload_bytes=max_payload_bytes,
        )

    ordered = tuple(sorted(observations, key=_observation_order_key))
    usable_count = sum(1 for observation in ordered if observation.is_usable)
    planned = usable_count * palette_size

    if observation_count > max_observations:
        return _result(
            status=REFUSED,
            reason=BUDGET,
            budget=MAX_OBSERVATIONS,
            proposals=(),
            observation_count=observation_count,
            palette_size=palette_size,
            candidate_evaluations=0,
            max_cost=float(max_cost),
            min_margin=float(min_margin),
            max_observations=max_observations,
            max_candidate_evaluations=max_candidate_evaluations,
            max_payload_bytes=max_payload_bytes,
        )
    if planned > max_candidate_evaluations:
        return _result(
            status=REFUSED,
            reason=BUDGET,
            budget=MAX_CANDIDATE_EVALUATIONS,
            proposals=(),
            observation_count=observation_count,
            palette_size=palette_size,
            candidate_evaluations=0,
            max_cost=float(max_cost),
            min_margin=float(min_margin),
            max_observations=max_observations,
            max_candidate_evaluations=max_candidate_evaluations,
            max_payload_bytes=max_payload_bytes,
        )

    palette_digest = _palette_digest(palette_items)
    scores = _score_observations(ordered, palette_items)

    groups: dict[BlockCoordinate, list[tuple[FaceObservation, FaceScore | None]]] = {}
    for observation, score in zip(ordered, scores):
        groups.setdefault(observation.cell, []).append((observation, score))

    proposals = tuple(
        _propose_cell(
            cell,
            groups[cell],
            palette_items,
            palette_digest,
            max_cost=float(max_cost),
            min_margin=float(min_margin),
        )
        for cell in sorted(groups)
    )
    status = PROPOSED if any(proposal.proposed for proposal in proposals) else REVIEW

    sized = _result(
        status=status,
        reason=None,
        budget=None,
        proposals=proposals,
        observation_count=observation_count,
        palette_size=palette_size,
        candidate_evaluations=planned,
        max_cost=float(max_cost),
        min_margin=float(min_margin),
        max_observations=max_observations,
        max_candidate_evaluations=max_candidate_evaluations,
        max_payload_bytes=max_payload_bytes,
    )
    if sized.payload_bytes > max_payload_bytes:
        return _result(
            status=REFUSED,
            reason=BUDGET,
            budget=MAX_PAYLOAD_BYTES,
            proposals=(),
            observation_count=observation_count,
            palette_size=palette_size,
            candidate_evaluations=planned,
            max_cost=float(max_cost),
            min_margin=float(min_margin),
            max_observations=max_observations,
            max_candidate_evaluations=max_candidate_evaluations,
            max_payload_bytes=max_payload_bytes,
        )
    return sized


__all__ = [
    "ABSTENTION_REASONS",
    "AppearanceStateProposalError",
    "BUDGET",
    "BUDGETS",
    "CONFLICTING_FACES",
    "EMPTY_PALETTE",
    "FaceStateProposal",
    "FaceStateProposalResult",
    "HIGH_COST",
    "MAX_CANDIDATE_EVALUATIONS",
    "MAX_OBSERVATIONS",
    "MAX_PAYLOAD_BYTES",
    "NO_OBSERVATION",
    "PRODUCER_VERSION",
    "PROPOSAL_SCHEMA",
    "PROPOSED",
    "REFUSED",
    "RESULT_SCHEMA",
    "REVIEW",
    "WEAK_MARGIN",
    "propose_face_states",
]
