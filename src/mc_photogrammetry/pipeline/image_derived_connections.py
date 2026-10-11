"""Bounded M4 connected-camera evidence adapter over an image-derived session.

The accepted :mod:`mc_photogrammetry.pipeline.image_derived_session` establishes
*per-view* camera facts: one yaw-modulo-90 pose, explicit intrinsics, and the
explicitly unresolved position / scale / yaw-quadrant notes - and nothing
between views. This module is the narrow seam that asks the already-frozen
:func:`joint_matching.match_and_solve_pair` matcher what each unordered pair of a
session's frames establishes, and reports the answer as typed, digest-bound
evidence. It owns exactly four things:

* a deterministic enumeration of the session's unordered frame pairs, in the
  session's declared frame order;
* one delegated matcher call per pair, using the session's own
  :class:`~mc_photogrammetry.pipeline.contracts.CameraEstimate` values and its
  immutable, non-reopenable RGB/gray arrays, with the matcher's typed refusal
  reasons passed through unchanged;
* an immutable :class:`RelativeCameraEvidence` record per pair - pair ids,
  status/reason, the matcher's solved unit baseline direction *only* when the
  matcher solved, the relative rotation and yaw-class diagnostics, match/support
  counters, a deterministic match digest, and a deterministic evidence digest
  that binds the record to ``session_sha256``;
* an immutable :class:`ImageDerivedConnectionGraph` result with the canonical
  pair ordering, the solved connected components, the declared pair and total
  budgets, and a deterministic graph digest.

The adapter never composes absolute camera translations, never infers or
exports metric scale, never derives bounds, scene transforms, block volumes or
face observations, and has no export/CLI/UI/manual-metadata surface. A solved
pair contributes exactly one edge ``left -> right`` with the matcher's unit
baseline direction and relative rotation; the absolute positions of the two
cameras remain unresolved, and the session's unresolved facts are copied into
the result untouched.

Failure is atomic. Every session source binding is validated - schema and
version, frame/camera/local-camera key order, forbidden position/scale claims,
the retained unresolved facts, the session digest, and each frame's immutable,
non-reopenable `uint8` pixels and pixel digest - before the pair budget is
consulted, so a malformed, tampered or pixel-less session is refused as its own
typed reason and never as ``connections_budget_exhausted``. A read-only but
numpy-owned (reopenable) buffer is rejected by the same non-reopenable-base
invariant the accepted session module uses, so a swapped buffer cannot reach the
matcher. The pair budget is checked before any matcher call; the total support
budget is enforced while evidence is still private, so crossing it returns a
refusal with no evidence and no components rather than a partial graph. The
three data refusals carry a typed ``reason`` on the graph result; a caller that
passes something other than a session or a matcher, or a matcher that violates
its own contract (including a malformed hypothesis diagnostic), raises
:class:`ImageDerivedConnectionError`.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from mc_photogrammetry.pipeline.image_derived_session import (
    ImageDerivedSession,
    ImageDerivedSessionError,
    # The accepted session module owns the non-reopenable-pixel invariant; the
    # adapter reuses that exact predicate rather than restating it, so the two
    # seams cannot drift apart.
    _has_immutable_bytes_base,
    image_derived_session_from_record,
    image_derived_session_to_record,
    image_sha256,
)
from mc_photogrammetry.pipeline.joint_matching import (
    JOINT_REFUSAL_REASONS,
    JointPairResult,
    match_and_solve_pair,
)

__all__ = [
    "IMAGE_DERIVED_CONNECTIONS_SCHEMA",
    "IMAGE_DERIVED_CONNECTIONS_VERSION",
    "IMAGE_DERIVED_CONNECTIONS_PRODUCER",
    "CONNECTIONS_INPUT_INVALID",
    "CONNECTIONS_MATCHER_MISMATCH",
    "CONNECTIONS_NO_PIXELS",
    "CONNECTIONS_SESSION_MISMATCH",
    "CONNECTIONS_BUDGET_EXHAUSTED",
    "CONNECTIONS_INVALID_BASELINE_DIRECTION",
    "CONNECTIONS_REFUSAL_REASONS",
    "CONNECTIONS_PAIR_REFUSAL_REASONS",
    "ImageDerivedConnectionError",
    "ConnectionBudgets",
    "YawClassDiagnostic",
    "RelativeCameraEvidence",
    "ImageDerivedConnectionGraph",
    "make_connection_budgets",
    "build_image_derived_connections",
]

IMAGE_DERIVED_CONNECTIONS_SCHEMA = "mc-photogrammetry-image-derived-connections"
"""Explicit graph-result schema name."""

IMAGE_DERIVED_CONNECTIONS_VERSION = 1
"""Explicit graph-result version; an unknown value is refused, never guessed."""

IMAGE_DERIVED_CONNECTIONS_PRODUCER = "m4-image-derived-connections-1"
"""Producer identity recorded in the report and carried by this module."""

CONNECTIONS_INPUT_INVALID = "connections_input_invalid"
"""Raised reason: the call itself was malformed (wrong type, unusable matcher)."""

CONNECTIONS_MATCHER_MISMATCH = "connections_matcher_mismatch"
"""Raised reason: a matcher return value does not satisfy its own contract."""

CONNECTIONS_NO_PIXELS = "connections_no_pixels"
"""Graph refusal: at least one frame carries no pixels (e.g. a record rebuild)."""

CONNECTIONS_SESSION_MISMATCH = "connections_session_mismatch"
"""Graph refusal: a session source binding or pixel digest does not replay."""

CONNECTIONS_BUDGET_EXHAUSTED = "connections_budget_exhausted"
"""Graph refusal: the pair budget or the total support budget was exceeded."""

CONNECTIONS_INVALID_BASELINE_DIRECTION = "invalid_baseline_direction"
"""Per-pair refusal: the matcher solved, but its baseline is not a unit vector."""

CONNECTIONS_REFUSAL_REASONS: tuple[str, ...] = (
    CONNECTIONS_NO_PIXELS,
    CONNECTIONS_SESSION_MISMATCH,
    CONNECTIONS_BUDGET_EXHAUSTED,
)
"""Every typed ``reason`` an :class:`ImageDerivedConnectionGraph` may report."""

CONNECTIONS_PAIR_REFUSAL_REASONS: tuple[str, ...] = (
    JOINT_REFUSAL_REASONS + (CONNECTIONS_INVALID_BASELINE_DIRECTION,)
)
"""Every typed ``reason`` a refused :class:`RelativeCameraEvidence` may carry.

A pair refusal is either the matcher's own typed token (passed through
verbatim, never renamed) or this adapter's single local guard for a matcher that
claims a solve while reporting a non-unit baseline direction.
"""

_TINY = 1e-12
_UNIT_TOLERANCE = 1e-6
_MATCH_DIGEST_DOMAIN = b"mc-photogrammetry-image-derived-connections-matches-v1"


class ImageDerivedConnectionError(ValueError):
    """A malformed call or a matcher return that violates its contract.

    ``reason`` is :data:`CONNECTIONS_INPUT_INVALID` or
    :data:`CONNECTIONS_MATCHER_MISMATCH`; ``detail`` is a human diagnostic that
    never replaces the typed reason. Data conditions the caller can receive as a
    typed graph refusal are *not* raised.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(reason if not detail else f"{reason}: {detail}")


@dataclass(frozen=True, slots=True)
class ConnectionBudgets:
    """Explicit pair and total limits for one graph build; both are declared.

    ``max_pairs`` is the pair budget: the number of unordered frame pairs the
    adapter may evaluate. It is checked against ``C(n, 2)`` *before* any matcher
    call, so a session over the pair budget performs no work and reports no
    evidence. ``max_total_support`` is the total budget: the sum of every
    evaluated pair's selected support. It is enforced while the evidence list is
    still private, so crossing it refuses the whole graph instead of emitting a
    prefix. A limit of ``0`` is legal and deliberately strict.
    """

    max_pairs: int
    max_total_support: int
    budget_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.max_pairs) is not int or self.max_pairs < 0:
            raise ValueError("max_pairs must be a non-negative integer")
        if type(self.max_total_support) is not int or self.max_total_support < 0:
            raise ValueError("max_total_support must be a non-negative integer")
        expected = _digest(_budget_payload(self))
        if self.budget_sha256 is None:
            object.__setattr__(self, "budget_sha256", expected)
        elif not _is_sha256(self.budget_sha256) or self.budget_sha256 != expected:
            raise ValueError("connection budget digest mismatch")

    def to_payload(self) -> dict[str, Any]:
        """Canonical budget payload, digest included."""
        payload = _budget_payload(self)
        payload["budget_sha256"] = self.budget_sha256
        return payload


def make_connection_budgets(*, max_pairs: int, max_total_support: int) -> ConnectionBudgets:
    """Build a self-validating budget from explicit caps (no hidden defaults)."""
    return ConnectionBudgets(max_pairs=max_pairs, max_total_support=max_total_support)


@dataclass(frozen=True, slots=True)
class YawClassDiagnostic:
    """One ranked yaw-quadrant hypothesis, reduced to immutable diagnostics."""

    yaw_class: int
    support: int
    spatial_bins: int
    median_selected_residual_degrees: float | None
    parallax_degrees: float | None
    cheirality_fraction: float | None
    singular_ratio: float | None
    passed: bool
    refusal: str

    def __post_init__(self) -> None:
        if (type(self.yaw_class) is not int or not 0 <= self.yaw_class <= 3
                or type(self.support) is not int or self.support < 0
                or type(self.spatial_bins) is not int or self.spatial_bins < 0):
            raise ValueError("yaw diagnostic counts must be non-negative integers")
        if type(self.passed) is not bool or not isinstance(self.refusal, str):
            raise ValueError("yaw diagnostic flags must be a boolean and a string")
        for name in ("median_selected_residual_degrees", "parallax_degrees",
                     "cheirality_fraction", "singular_ratio"):
            _require_finite_or_none(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class RelativeCameraEvidence:
    """One pair's immutable, digest-bound relative-camera evidence.

    ``status`` is ``solved`` or ``refused``. On a solve, ``baseline_direction``
    is the matcher's unit world baseline direction (left eye to right eye, in
    the left estimate's anchor frame), ``relative_rotation`` is its 3x3 relative
    rotation, and ``right_yaw_class`` is the recovered right-view quadrant; on a
    refusal all three are ``None`` and ``reason`` is a
    :data:`CONNECTIONS_PAIR_REFUSAL_REASONS` token. No absolute translation,
    scale or scene transform is present. ``matches_sha256`` digests the pair's
    selected one-to-one match array; ``evidence_sha256`` digests this record and
    binds it to ``session_sha256``.
    """

    left_id: str
    right_id: str
    status: str
    reason: str
    baseline_direction: tuple[float, float, float] | None
    right_yaw_class: int | None
    relative_rotation: tuple[tuple[float, float, float], ...] | None
    match_count: int
    support: int
    support_floor: int
    raw_keypoints: tuple[int, int]
    candidate_edges: int
    mutual_survivors: int
    median_selected_residual_degrees: float | None
    parallax_degrees: float | None
    cheirality_fraction: float | None
    singular_ratio: float | None
    ambiguity_margin: int | None
    hypotheses: tuple[YawClassDiagnostic, ...]
    matches_sha256: str
    session_sha256: str
    evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.left_id, str) or not self.left_id
                or not isinstance(self.right_id, str) or not self.right_id
                or self.left_id == self.right_id):
            raise ValueError("pair ids must be two distinct non-empty strings")
        vector_fields = (self.baseline_direction, self.relative_rotation, self.right_yaw_class)
        if self.status == "solved":
            if self.reason != "":
                raise ValueError("solved evidence cannot carry a refusal reason")
            if any(field is None for field in vector_fields):
                raise ValueError(
                    "solved evidence must carry a baseline direction, rotation and yaw class")
        elif self.status == "refused":
            if self.reason not in CONNECTIONS_PAIR_REFUSAL_REASONS:
                raise ValueError("refused evidence needs a typed refusal reason")
            if any(field is not None for field in vector_fields):
                raise ValueError(
                    "refused evidence cannot carry a solved direction, rotation or yaw class")
        else:
            raise ValueError("evidence status must be solved or refused")
        if self.baseline_direction is not None:
            _require_unit_direction(self.baseline_direction)
        if self.relative_rotation is not None:
            _require_rotation_matrix(self.relative_rotation)
        for name in ("match_count", "support", "support_floor", "candidate_edges",
                     "mutual_survivors"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (not isinstance(self.raw_keypoints, tuple) or len(self.raw_keypoints) != 2
                or any(type(value) is not int or value < 0 for value in self.raw_keypoints)):
            raise ValueError("raw_keypoints must be two non-negative integers")
        for name in ("median_selected_residual_degrees", "parallax_degrees",
                     "cheirality_fraction", "singular_ratio"):
            _require_finite_or_none(getattr(self, name), name)
        if self.ambiguity_margin is not None and type(self.ambiguity_margin) is not int:
            raise ValueError("ambiguity_margin must be an integer or None")
        if (not isinstance(self.hypotheses, tuple)
                or any(not isinstance(item, YawClassDiagnostic) for item in self.hypotheses)):
            raise ValueError("hypotheses must be a tuple of yaw diagnostics")
        if not _is_sha256(self.matches_sha256):
            raise ValueError("match digest must be a lowercase SHA-256 hex string")
        if not _is_sha256(self.session_sha256):
            raise ValueError("session digest must be a lowercase SHA-256 hex string")
        expected = _digest(_evidence_payload(self))
        if self.evidence_sha256 is None:
            object.__setattr__(self, "evidence_sha256", expected)
        elif not _is_sha256(self.evidence_sha256) or self.evidence_sha256 != expected:
            raise ValueError("evidence digest mismatch")

    def to_payload(self) -> dict[str, Any]:
        """Canonical evidence payload, digest included."""
        payload = _evidence_payload(self)
        payload["evidence_sha256"] = self.evidence_sha256
        return payload


@dataclass(frozen=True, slots=True)
class ImageDerivedConnectionGraph:
    """The deterministic connected-camera evidence graph for one session.

    An ``ok`` graph carries exactly one :class:`RelativeCameraEvidence` per
    unordered frame pair in canonical order and the solved connected components
    over all frames (isolated frames included as singletons). A ``refused`` graph
    carries a typed :data:`CONNECTIONS_REFUSAL_REASONS` reason, no evidence and
    no components - never a partial result. ``scale_known`` is hard-frozen
    ``False`` and ``unresolved`` is the session's unresolved-fact tuple, so the
    unresolved scale/translation/yaw facts survive the adapter.
    """

    schema: str
    version: int
    status: str
    reason: str
    detail: str
    session_sha256: str
    budgets: ConnectionBudgets
    frame_ids: tuple[str, ...]
    pair_count: int
    evidence: tuple[RelativeCameraEvidence, ...]
    components: tuple[tuple[str, ...], ...]
    unresolved: tuple[str, ...]
    scale_known: bool
    graph_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema != IMAGE_DERIVED_CONNECTIONS_SCHEMA:
            raise ValueError("unsupported connections schema")
        if type(self.version) is not int or self.version != IMAGE_DERIVED_CONNECTIONS_VERSION:
            raise ValueError("unsupported connections version")
        if self.status not in ("ok", "refused"):
            raise ValueError("graph status must be ok or refused")
        if not isinstance(self.budgets, ConnectionBudgets):
            raise ValueError("graph budgets must be ConnectionBudgets")
        if (not isinstance(self.frame_ids, tuple)
                or any(not isinstance(name, str) or not name for name in self.frame_ids)
                or len(set(self.frame_ids)) != len(self.frame_ids)):
            raise ValueError("frame ids must be a tuple of unique non-empty strings")
        if type(self.pair_count) is not int or self.pair_count != _pair_count(len(self.frame_ids)):
            raise ValueError("pair count must equal C(frames, 2)")
        if self.scale_known is not False:
            raise ValueError("an image-derived graph cannot claim metric scale")
        if (not isinstance(self.unresolved, tuple)
                or any(not isinstance(note, str) for note in self.unresolved)):
            raise ValueError("unresolved notes must be a tuple of strings")
        if not isinstance(self.session_sha256, str):
            raise ValueError("session digest must be a string")
        if not isinstance(self.detail, str):
            raise ValueError("refusal detail must be a string")
        if (not isinstance(self.evidence, tuple)
                or any(not isinstance(record, RelativeCameraEvidence)
                       for record in self.evidence)):
            raise ValueError("evidence must be a tuple of evidence records")
        if self.status == "ok":
            if self.reason != "":
                raise ValueError("an ok graph cannot carry a refusal reason")
            if not _is_sha256(self.session_sha256):
                raise ValueError("an ok graph needs the verified session digest")
            if not self.frame_ids:
                raise ValueError("an ok graph needs at least one frame")
            if len(self.evidence) != self.pair_count:
                raise ValueError("an ok graph carries exactly one record per pair")
            expected_pairs = _ordered_pairs(self.frame_ids)
            if tuple((record.left_id, record.right_id)
                     for record in self.evidence) != expected_pairs:
                raise ValueError("evidence is not in canonical pair order")
            if self.components != _solved_components(self.frame_ids, self.evidence):
                raise ValueError("components do not replay from the solved evidence")
        else:
            if self.reason not in CONNECTIONS_REFUSAL_REASONS:
                raise ValueError("a refused graph needs a typed refusal reason")
            if self.evidence != () or self.components != ():
                raise ValueError("a refused graph must not carry a partial result")
        expected = _digest(_graph_payload(self))
        if self.graph_sha256 is None:
            object.__setattr__(self, "graph_sha256", expected)
        elif not _is_sha256(self.graph_sha256) or self.graph_sha256 != expected:
            raise ValueError("graph digest mismatch")

    def to_record(self) -> dict[str, Any]:
        """Canonical graph record, digest included."""
        record = _graph_payload(self)
        record["graph_sha256"] = self.graph_sha256
        return record


@dataclass(frozen=True, slots=True)
class _PairCorrespondenceObservation:
    """Private diagnostic snapshot; never serialized into the production graph.

    Coordinates are copied before matcher conversion, capped at 64 rows, and
    delivered only after the ordinary pair contract and support budget pass.
    A truncated snapshot cannot establish union consistency.
    """

    left_id: str
    right_id: str
    session_sha256: str
    matches_sha256: str
    evidence_sha256: str
    status: str
    reason: str
    parallax_degrees: float | None
    ambiguity_margin: int | None
    source_count: int
    complete: bool
    matches: tuple[tuple[float, float, float, float], ...]
    left_pixel_sha256: str = ""
    right_pixel_sha256: str = ""
    left_shape: tuple[int, ...] = ()
    right_shape: tuple[int, ...] = ()


def build_image_derived_connections(
    session: ImageDerivedSession,
    *,
    budgets: ConnectionBudgets,
    matcher: Callable[..., JointPairResult] = match_and_solve_pair,
    _correspondence_observer=None,
) -> ImageDerivedConnectionGraph:
    """Evaluate every unordered session pair and emit the evidence graph.

    All session source bindings are validated first, then the pair budget, then
    the delegated matcher is called once per pair in canonical order with the
    session's own ``CameraEstimate`` values and immutable images. A pair's
    matcher refusal is preserved as a typed record; a solve is preserved as a
    unit direction plus its rotation/yaw diagnostics. Exceeding the total
    support budget returns a refusal with no evidence. ``matcher`` defaults to
    the production :func:`joint_matching.match_and_solve_pair` and exists as the
    injectable test seam.
    """
    if not isinstance(session, ImageDerivedSession):
        raise ImageDerivedConnectionError(
            CONNECTIONS_INPUT_INVALID, "expected an accepted ImageDerivedSession")
    if not isinstance(budgets, ConnectionBudgets):
        raise ImageDerivedConnectionError(
            CONNECTIONS_INPUT_INVALID, "expected a ConnectionBudgets")
    if not callable(matcher):
        raise ImageDerivedConnectionError(
            CONNECTIONS_INPUT_INVALID, "matcher must be callable")
    if _correspondence_observer is not None and not callable(_correspondence_observer):
        raise ImageDerivedConnectionError(CONNECTIONS_INPUT_INVALID,
                                          "diagnostic observer must be callable")

    frame_ids = _safe_frame_ids(session)
    declared_digest = session.session_sha256 if isinstance(session.session_sha256, str) else ""

    invalid = _validate_session(session)
    if invalid is not None:
        reason, detail = invalid
        return _refused_graph(
            session_sha256=declared_digest,
            frame_ids=frame_ids,
            budgets=budgets,
            reason=reason,
            detail=detail,
            unresolved=_safe_unresolved(session),
        )

    pairs = _ordered_pairs(frame_ids)
    if len(pairs) > budgets.max_pairs:
        return _refused_graph(
            session_sha256=session.session_sha256,
            frame_ids=frame_ids,
            budgets=budgets,
            reason=CONNECTIONS_BUDGET_EXHAUSTED,
            detail=f"unordered pair count {len(pairs)} exceeds max_pairs "
                   f"{budgets.max_pairs}",
            unresolved=tuple(session.unresolved),
        )

    by_id = {frame.frame_id: frame for frame in session.frames}
    evidence: list[RelativeCameraEvidence] = []
    total_support = 0
    for left_id, right_id in pairs:
        result = _call_matcher(matcher, session, by_id, left_id, right_id)
        # No extra array conversion, copy or callback on the ordinary path.
        snapshot = None
        if _correspondence_observer is not None:
            try:
                values = result.matches
                snapshot = tuple(tuple(float(row[i]) for i in range(4))
                                 for row in values[:64] if len(row) == 4)
            except (TypeError, ValueError, AttributeError):
                snapshot = ()
        try:
            record = _evidence_from_result(result, left_id, right_id, session.session_sha256)
        except ImageDerivedConnectionError:
            raise
        except (TypeError, ValueError, KeyError, IndexError, AttributeError) as error:
            # Defensive seal: a matcher-derived field that fails construction or
            # validation must surface as a typed contract mismatch, never as a
            # raw ValueError from the record/diagnostic constructors.
            raise ImageDerivedConnectionError(
                CONNECTIONS_MATCHER_MISMATCH,
                f"matcher evidence for {left_id}->{right_id} is malformed: {error}",
            ) from error
        total_support += record.support
        if total_support > budgets.max_total_support:
            return _refused_graph(
                session_sha256=session.session_sha256,
                frame_ids=frame_ids,
                budgets=budgets,
                reason=CONNECTIONS_BUDGET_EXHAUSTED,
                detail=f"total support {total_support} exceeds max_total_support "
                       f"{budgets.max_total_support}",
                unresolved=tuple(session.unresolved),
            )
        evidence.append(record)
        if _correspondence_observer is not None:
            observation = _PairCorrespondenceObservation(
                left_id, right_id, session.session_sha256, record.matches_sha256,
                record.evidence_sha256, record.status, record.reason,
                record.parallax_degrees, record.ambiguity_margin, record.match_count,
                record.match_count <= 64 and len(snapshot) == record.match_count
                and all(len(row) == 4 and all(math.isfinite(v) for v in row)
                        for row in snapshot), snapshot,
                by_id[left_id].image_sha256, by_id[right_id].image_sha256,
                tuple(by_id[left_id].image_shape), tuple(by_id[right_id].image_shape))
            try:
                _correspondence_observer(observation)
            except Exception:
                # The collector owns incompleteness; telemetry cannot alter a
                # production solve, refusal, serialization or default gate.
                try:
                    _correspondence_observer.discard("observer_error")
                except Exception:
                    pass

    records = tuple(evidence)
    return ImageDerivedConnectionGraph(
        schema=IMAGE_DERIVED_CONNECTIONS_SCHEMA,
        version=IMAGE_DERIVED_CONNECTIONS_VERSION,
        status="ok",
        reason="",
        detail="",
        session_sha256=session.session_sha256,
        budgets=budgets,
        frame_ids=frame_ids,
        pair_count=len(pairs),
        evidence=records,
        components=_solved_components(frame_ids, records),
        unresolved=tuple(session.unresolved),
        scale_known=False,
    )


# --- Session validation -------------------------------------------------------

def _validate_session(session: ImageDerivedSession) -> tuple[str, str] | None:
    """Replay every source binding before any budget or matcher work.

    The accepted session deserializer is the binding authority: the session is
    serialized and rebuilt, so its exact key sets, schema/version, forbidden
    position/scale claims, retained unresolved facts and digest are re-checked
    by the module that owns them. Pixels are intentionally absent from a record,
    so each frame's immutable uint8 array and pixel digest are validated
    directly here. Returns a ``(reason, detail)`` data refusal, or ``None``.
    """
    try:
        record = image_derived_session_to_record(session)
        rebuilt = image_derived_session_from_record(record)
    except ImageDerivedSessionError as error:
        return (CONNECTIONS_SESSION_MISMATCH,
                f"session record does not replay: {error.reason}")
    except (TypeError, ValueError, KeyError, IndexError, AttributeError) as error:
        return (CONNECTIONS_SESSION_MISMATCH, f"session record is malformed: {error}")
    if rebuilt.session_sha256 != session.session_sha256:
        return (CONNECTIONS_SESSION_MISMATCH, "session digest does not bind")
    try:
        frames = tuple(session.frames)
        frame_ids = tuple(frame.frame_id for frame in frames)
        camera_ids = tuple(session.cameras)
        local_ids = tuple(session.local_cameras)
    except (TypeError, AttributeError):
        return (CONNECTIONS_SESSION_MISMATCH, "session bindings are malformed")
    if camera_ids != frame_ids or local_ids != frame_ids:
        return (CONNECTIONS_SESSION_MISMATCH,
                "camera maps must key every frame exactly once, in frame order")
    for frame in session.input_frames:
        problem = _frame_pixel_problem(frame)
        if problem is not None:
            return problem
    return None


def _frame_pixel_problem(frame: Any) -> tuple[str, str] | None:
    frame_id = getattr(frame, "frame_id", "?")
    image = getattr(frame, "image", None)
    if image is None:
        return (CONNECTIONS_NO_PIXELS, f"{frame_id}: frame carries no pixels")
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        return (CONNECTIONS_SESSION_MISMATCH, f"{frame_id}: pixels are not a uint8 array")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        return (CONNECTIONS_SESSION_MISMATCH, f"{frame_id}: pixels are not gray or RGB")
    if image.flags.writeable:
        return (CONNECTIONS_SESSION_MISMATCH, f"{frame_id}: pixels are writeable")
    # ``writeable=False`` alone is not immutability: a numpy-owned array can be
    # reopened with ``setflags(write=True)`` and then mutated under the matcher.
    # Only a buffer chain that terminates at immutable ``bytes`` is genuinely
    # non-reopenable, so require the session module's own invariant here too.
    if not _has_immutable_bytes_base(image):
        return (CONNECTIONS_SESSION_MISMATCH,
                f"{frame_id}: pixels are backed by reopenable memory, not immutable bytes")
    if tuple(image.shape) != tuple(frame.image_shape):
        return (CONNECTIONS_SESSION_MISMATCH, f"{frame_id}: pixel shape does not bind")
    if image_sha256(image) != frame.image_sha256:
        return (CONNECTIONS_SESSION_MISMATCH, f"{frame_id}: pixel digest does not bind")
    return None


# --- Matcher delegation and evidence construction -----------------------------

def _call_matcher(
    matcher: Callable[..., JointPairResult],
    session: ImageDerivedSession,
    by_id: dict[str, Any],
    left_id: str,
    right_id: str,
) -> JointPairResult:
    left = by_id[left_id]
    right = by_id[right_id]
    result = matcher(
        left_id, left.image, right_id, right.image,
        left_camera=session.cameras[left_id],
        right_camera=session.cameras[right_id],
    )
    if (getattr(result, "left_id", None) != left_id
            or getattr(result, "right_id", None) != right_id):
        raise ImageDerivedConnectionError(
            CONNECTIONS_MATCHER_MISMATCH, "matcher returned a different pair")
    return result


def _evidence_from_result(
    result: JointPairResult, left_id: str, right_id: str, session_sha256: str,
) -> RelativeCameraEvidence:
    status = getattr(result, "status", None)
    solved = bool(getattr(result, "solved", False))
    if status not in ("solved", "refused") or (status == "solved") != solved:
        raise ImageDerivedConnectionError(
            CONNECTIONS_MATCHER_MISMATCH, "matcher status does not bind to its solved flag")
    matches = _match_array(getattr(result, "matches", None))
    matches_sha256 = _array_sha256(matches, _MATCH_DIGEST_DOMAIN)

    baseline: tuple[float, float, float] | None = None
    rotation: tuple[tuple[float, float, float], ...] | None = None
    yaw_class: int | None = None
    reason = ""
    if solved:
        pose = getattr(result, "pose", None)
        if pose is None:
            raise ImageDerivedConnectionError(
                CONNECTIONS_MATCHER_MISMATCH, "a solved matcher result carries no pose")
        baseline = _unit_direction(getattr(pose, "baseline_direction", None))
        if baseline is None:
            status = "refused"
            reason = CONNECTIONS_INVALID_BASELINE_DIRECTION
        else:
            rotation = _rotation_tuple(getattr(pose, "relative_rotation", None))
            yaw_class = _yaw_class(getattr(result, "right_yaw_class", None))
            if rotation is None or yaw_class is None:
                raise ImageDerivedConnectionError(
                    CONNECTIONS_MATCHER_MISMATCH,
                    "a solved matcher result needs a rotation matrix and yaw class")
    else:
        reason = getattr(result, "reason", "")
        if reason not in JOINT_REFUSAL_REASONS:
            raise ImageDerivedConnectionError(
                CONNECTIONS_MATCHER_MISMATCH, "matcher refusal reason is not a typed token")

    return RelativeCameraEvidence(
        left_id=left_id,
        right_id=right_id,
        status=status,
        reason=reason,
        baseline_direction=baseline,
        right_yaw_class=yaw_class,
        relative_rotation=rotation,
        match_count=int(matches.shape[0]),
        support=_require_int(getattr(result, "support", None), "support"),
        support_floor=_require_int(getattr(result, "support_floor", None), "support_floor"),
        raw_keypoints=_int_pair(getattr(result, "raw_keypoints", None), "raw_keypoints"),
        candidate_edges=_require_int(getattr(result, "candidate_edges", None), "candidate_edges"),
        mutual_survivors=_require_int(
            getattr(result, "mutual_survivors", None), "mutual_survivors"),
        median_selected_residual_degrees=_optional_finite_matcher_field(
            getattr(result, "median_selected_residual_degrees", None),
            "median_selected_residual_degrees"),
        parallax_degrees=_optional_finite_matcher_field(
            getattr(result, "parallax_degrees", None), "parallax_degrees"),
        cheirality_fraction=_optional_finite_matcher_field(
            getattr(result, "cheirality_fraction", None), "cheirality_fraction"),
        singular_ratio=_optional_finite_matcher_field(
            getattr(result, "singular_ratio", None), "singular_ratio"),
        ambiguity_margin=_optional_int(getattr(result, "ambiguity_margin", None)),
        hypotheses=_hypotheses(getattr(result, "hypotheses", ())),
        matches_sha256=matches_sha256,
        session_sha256=session_sha256,
    )


def _hypotheses(values: Any) -> tuple[YawClassDiagnostic, ...]:
    if not isinstance(values, (tuple, list)):
        raise ImageDerivedConnectionError(
            CONNECTIONS_MATCHER_MISMATCH, "matcher hypotheses must be a sequence")
    return tuple(_yaw_diagnostic(value) for value in values)


def _yaw_diagnostic(value: Any) -> YawClassDiagnostic:
    """Reduce one matcher hypothesis, refusing every malformed field typed.

    ``yaw_class``, ``support``, ``spatial_bins``, ``passed`` and ``refusal`` are
    validated exactly - never coerced through ``int``/``bool``/``str`` - and the
    four optional diagnostics (``median_selected_residual_degrees``,
    ``parallax_degrees``, ``cheirality_fraction``, ``singular_ratio``) go through
    the strict optional-finite parser, so a malformed string/bool/non-finite
    value is refused typed instead of being laundered to ``None``. Any remaining
    construction error is wrapped as a matcher-contract error, so no raw
    ``ValueError`` escapes to the caller.
    """
    yaw_class = getattr(value, "yaw_class", None)
    if (isinstance(yaw_class, bool)
            or not isinstance(yaw_class, (int, np.integer))
            or not 0 <= int(yaw_class) <= 3):
        raise ImageDerivedConnectionError(
            CONNECTIONS_MATCHER_MISMATCH,
            "matcher hypothesis yaw_class must be an integer in 0..3")
    passed = getattr(value, "passed", None)
    if type(passed) is not bool:
        raise ImageDerivedConnectionError(
            CONNECTIONS_MATCHER_MISMATCH, "matcher hypothesis passed must be a boolean")
    refusal = getattr(value, "refusal", None)
    if not isinstance(refusal, str):
        raise ImageDerivedConnectionError(
            CONNECTIONS_MATCHER_MISMATCH, "matcher hypothesis refusal must be a string")
    support = _require_int(getattr(value, "support", None), "hypothesis support")
    spatial_bins = _require_int(
        getattr(value, "spatial_bins", None), "hypothesis spatial_bins")
    median_selected_residual_degrees = _optional_finite_matcher_field(
        getattr(value, "median_selected_residual_degrees", None),
        "hypothesis median_selected_residual_degrees")
    parallax_degrees = _optional_finite_matcher_field(
        getattr(value, "parallax_degrees", None), "hypothesis parallax_degrees")
    cheirality_fraction = _optional_finite_matcher_field(
        getattr(value, "cheirality_fraction", None), "hypothesis cheirality_fraction")
    singular_ratio = _optional_finite_matcher_field(
        getattr(value, "singular_ratio", None), "hypothesis singular_ratio")
    try:
        return YawClassDiagnostic(
            yaw_class=int(yaw_class),
            support=support,
            spatial_bins=spatial_bins,
            median_selected_residual_degrees=median_selected_residual_degrees,
            parallax_degrees=parallax_degrees,
            cheirality_fraction=cheirality_fraction,
            singular_ratio=singular_ratio,
            passed=passed,
            refusal=refusal,
        )
    except (TypeError, ValueError) as error:
        raise ImageDerivedConnectionError(
            CONNECTIONS_MATCHER_MISMATCH,
            f"matcher hypothesis diagnostic is malformed: {error}") from error


def _match_array(value: Any) -> np.ndarray:
    if value is None:
        raise ImageDerivedConnectionError(
            CONNECTIONS_MATCHER_MISMATCH, "matcher returned no match array")
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ImageDerivedConnectionError(
            CONNECTIONS_MATCHER_MISMATCH, "matcher matches must be an N x 4 array")
    return array


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ImageDerivedConnectionError(
            CONNECTIONS_MATCHER_MISMATCH, f"matcher {name} is not an integer")
    result = int(value)
    if result < 0:
        raise ImageDerivedConnectionError(
            CONNECTIONS_MATCHER_MISMATCH, f"matcher {name} is negative")
    return result


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return _require_int(value, "count")


def _int_pair(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ImageDerivedConnectionError(
            CONNECTIONS_MATCHER_MISMATCH, f"matcher {name} must be a pair")
    return (_require_int(value[0], name), _require_int(value[1], name))


def _unit_direction(value: Any) -> tuple[float, float, float] | None:
    """Accept a finite unit 3-vector; return ``None`` for anything else."""
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        return None
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= _TINY or abs(norm - 1.0) > _UNIT_TOLERANCE:
        return None
    return (float(array[0]), float(array[1]), float(array[2]))


def _rotation_tuple(value: Any) -> tuple[tuple[float, float, float], ...] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        return None
    return tuple(tuple(float(entry) for entry in row) for row in array)


def _yaw_class(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return None
    result = int(value)
    return result if 0 <= result <= 3 else None


def _optional_finite_matcher_field(value: Any, name: str) -> float | None:
    """Strict optional-finite matcher numeric; never silently launders.

    ``None`` is the matcher's legitimate "not measured" value and is preserved
    exactly. Otherwise the value must be an exact real numeric - a Python
    ``int``/``float`` or a NumPy integer/floating scalar, never a ``bool`` (and
    never a NumPy boolean) - and must be finite. A string, a boolean, ``NaN``,
    ``inf`` or any other object is malformed and raises the typed
    :class:`ImageDerivedConnectionError` instead of being quietly reduced to
    ``None``. Keeping every malformed diagnostic loud is what lets the graph
    builder refuse a non-conforming matcher rather than record fabricated
    "unmeasured" evidence.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise ImageDerivedConnectionError(
            CONNECTIONS_MATCHER_MISMATCH,
            f"matcher {name} must be a finite number or None")
    result = float(value)
    if not math.isfinite(result):
        raise ImageDerivedConnectionError(
            CONNECTIONS_MATCHER_MISMATCH, f"matcher {name} must be finite")
    return result


# --- Canonical ordering, components, digests and payloads ---------------------

def _pair_count(frame_count: int) -> int:
    return frame_count * (frame_count - 1) // 2


def _ordered_pairs(frame_ids: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (frame_ids[left], frame_ids[right])
        for left in range(len(frame_ids))
        for right in range(left + 1, len(frame_ids))
    )


def _solved_components(
    frame_ids: tuple[str, ...], evidence: tuple[RelativeCameraEvidence, ...],
) -> tuple[tuple[str, ...], ...]:
    """Union-find over solved edges; components ordered by first frame."""
    parent = list(range(len(frame_ids)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    index_of = {frame_id: index for index, frame_id in enumerate(frame_ids)}
    for record in evidence:
        if record.status != "solved":
            continue
        left = find(index_of[record.left_id])
        right = find(index_of[record.right_id])
        if left != right:
            parent[max(left, right)] = min(left, right)
    groups: dict[int, list[str]] = {}
    for index, frame_id in enumerate(frame_ids):
        groups.setdefault(find(index), []).append(frame_id)
    return tuple(tuple(members) for _root, members in sorted(groups.items()))


def _safe_frame_ids(session: ImageDerivedSession) -> tuple[str, ...]:
    frames = getattr(session, "frames", None)
    if not isinstance(frames, tuple):
        return ()
    frame_ids: list[str] = []
    for frame in frames:
        frame_id = getattr(frame, "frame_id", None)
        if not isinstance(frame_id, str) or not frame_id:
            return ()
        frame_ids.append(frame_id)
    return tuple(frame_ids)


def _safe_unresolved(session: ImageDerivedSession) -> tuple[str, ...]:
    value = getattr(session, "unresolved", ())
    if isinstance(value, tuple) and all(isinstance(note, str) for note in value):
        return value
    return ()


def _refused_graph(
    *,
    session_sha256: str,
    frame_ids: tuple[str, ...],
    budgets: ConnectionBudgets,
    reason: str,
    detail: str,
    unresolved: tuple[str, ...],
) -> ImageDerivedConnectionGraph:
    return ImageDerivedConnectionGraph(
        schema=IMAGE_DERIVED_CONNECTIONS_SCHEMA,
        version=IMAGE_DERIVED_CONNECTIONS_VERSION,
        status="refused",
        reason=reason,
        detail=detail,
        session_sha256=session_sha256,
        budgets=budgets,
        frame_ids=frame_ids,
        pair_count=_pair_count(len(frame_ids)),
        evidence=(),
        components=(),
        unresolved=unresolved,
        scale_known=False,
    )


def _budget_payload(budgets: ConnectionBudgets) -> dict[str, Any]:
    return {
        "max_pairs": int(budgets.max_pairs),
        "max_total_support": int(budgets.max_total_support),
    }


def _hypothesis_payload(value: YawClassDiagnostic) -> dict[str, Any]:
    return {
        "yaw_class": value.yaw_class,
        "support": value.support,
        "spatial_bins": value.spatial_bins,
        "median_selected_residual_degrees": value.median_selected_residual_degrees,
        "parallax_degrees": value.parallax_degrees,
        "cheirality_fraction": value.cheirality_fraction,
        "singular_ratio": value.singular_ratio,
        "passed": bool(value.passed),
        "refusal": value.refusal,
    }


def _evidence_payload(evidence: RelativeCameraEvidence) -> dict[str, Any]:
    return {
        "left_id": evidence.left_id,
        "right_id": evidence.right_id,
        "status": evidence.status,
        "reason": evidence.reason,
        "baseline_direction": (None if evidence.baseline_direction is None
                               else list(evidence.baseline_direction)),
        "right_yaw_class": evidence.right_yaw_class,
        "relative_rotation": (None if evidence.relative_rotation is None
                              else [list(row) for row in evidence.relative_rotation]),
        "match_count": evidence.match_count,
        "support": evidence.support,
        "support_floor": evidence.support_floor,
        "raw_keypoints": list(evidence.raw_keypoints),
        "candidate_edges": evidence.candidate_edges,
        "mutual_survivors": evidence.mutual_survivors,
        "median_selected_residual_degrees": evidence.median_selected_residual_degrees,
        "parallax_degrees": evidence.parallax_degrees,
        "cheirality_fraction": evidence.cheirality_fraction,
        "singular_ratio": evidence.singular_ratio,
        "ambiguity_margin": evidence.ambiguity_margin,
        "hypotheses": [_hypothesis_payload(item) for item in evidence.hypotheses],
        "matches_sha256": evidence.matches_sha256,
        "session_sha256": evidence.session_sha256,
    }


def _graph_payload(graph: ImageDerivedConnectionGraph) -> dict[str, Any]:
    return {
        "schema": graph.schema,
        "version": graph.version,
        "status": graph.status,
        "reason": graph.reason,
        "detail": graph.detail,
        "session_sha256": graph.session_sha256,
        "budgets": _budget_payload(graph.budgets),
        "frame_ids": list(graph.frame_ids),
        "pair_count": graph.pair_count,
        "evidence": [
            {**_evidence_payload(record), "evidence_sha256": record.evidence_sha256}
            for record in graph.evidence
        ],
        "components": [list(component) for component in graph.components],
        "unresolved": list(graph.unresolved),
        "scale_known": bool(graph.scale_known),
    }


def _array_sha256(array: np.ndarray, domain: bytes) -> str:
    """Domain-separated SHA-256 over an array's rank, shape, dtype and bytes."""
    framed = [domain, bytes([array.ndim])]
    for dimension in array.shape:
        framed.append(int(dimension).to_bytes(8, "little", signed=False))
    framed.append(array.dtype.str.encode("ascii"))
    framed.append(np.ascontiguousarray(array).tobytes(order="C"))
    return hashlib.sha256(b"".join(framed)).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _require_finite_or_none(value: Any, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number or None")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _require_unit_direction(value: Any) -> None:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError("baseline direction must be three finite numbers")
    norm = float(np.linalg.norm(array))
    if norm <= _TINY or abs(norm - 1.0) > _UNIT_TOLERANCE:
        raise ValueError("baseline direction must be a unit vector")


def _require_rotation_matrix(value: Any) -> None:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ValueError("relative rotation must be a finite 3x3 matrix")
