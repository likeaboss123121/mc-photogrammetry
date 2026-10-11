"""Evaluation-only image-only route frontier records (A1).

This module is the accepted A1 implementation of
``docs/reports/image-only-loss-attribution-design-2026-10-03.md`` section 5
(Tranche A1).  It is **frontier/status measurement only**: it reports which
checkpoints of the two image-only routes were reached, the typed reason the
first one was not, and nothing that could be read as visible accuracy.

Two routes are measured separately and never pooled:

* **Route A** - the normal-mode single-view ``reconstruct`` facade
  (``admission`` -> ``camera`` -> ``facade_grid`` -> ``palette_label`` ->
  ``terminal``).
* **Route B** - the intended multiview ``image-derived-output`` production chain
  (``entry`` -> ``lattice`` -> ``geometry`` -> ``output``).

Contract
--------

* **Pure.**  This module performs no filesystem I/O, imports no production
  orchestration (``cli``, ``project.store``, ``pipeline.image_derived_production``),
  no exporter and no reconstruction body.  It reads only fields the accepted
  in-memory records already expose, so production cannot reach it and it cannot
  reach production's manual metadata.
* **Five labelled spaces.**  Every checkpoint names one fixed universe: geometry
  ``G``, face/pixel ownership ``O``, pair correspondence ``P``, candidate/palette
  ``C``, and exact-state ``E``.  A camera checkpoint is a diagnostic with no cell
  denominator and no cell-loss equation.  Nothing here sums, subtracts or divides
  across universes; A1 carries no numerator or denominator at all, so the metric
  is always typed ``null`` and A2 owns the scoreable numbers.
* **Typed unavailable, never a measured zero.**  A checkpoint whose record was
  not carried is ``reachable=False`` with a non-empty ``blocked`` token; it
  carries no digest and contributes no count.  Once a checkpoint is unreachable,
  every later checkpoint in that route's canonical order is unreachable too.
* **Deterministic and digest-bound.**  Every record has a canonical JSON
  ``to_record()``; two builds over identical inputs give byte-identical JSON and
  digest.  A record that does not replay exactly raises
  :class:`FrontierRefusal` instead of building a partially trusted frontier.
* **Accepted digests bind, and stop is not terminal.**  When a route-B production
  result is supplied, the frontier takes its terminal ``stage``/``status``/
  ``reason`` from that validated record and proves the terminal stage's record was
  carried and binds its digest; the *first unattempted checkpoint* is reported
  separately.  A carried geometry record whose status is ``unavailable`` can
  therefore be the terminal failure while ``output`` is the first unattempted
  stage, so a frontier can never claim a terminal the returned record
  contradicts.

A1 deliberately produces no visible-truth number: the frozen evaluator truth,
masks, references and alignment belong to A2's evaluator-only builders, never to
these adapters.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

__all__ = [
    "FRONTIER_SCHEMA",
    "FRONTIER_PRODUCER",
    "ROUTE_A_STAGES",
    "ROUTE_B_STAGES",
    "FrontierRefusal",
    "RecordKind",
    "Universe",
    "Route",
    "FrontierCheckpoint",
    "FrontierTransition",
    "ImageOnlyFrontier",
    "RouteAFrontier",
    "route_a_frontier",
    "route_b_frontier",
    "frontier_unavailable",
    "first_observable_failure",
    "canonical_json",
    "record_digest",
]

FRONTIER_SCHEMA = "mc-photogrammetry-image-only-frontier-1"
"""Schema name carried by every canonical A1 frontier record."""

FRONTIER_PRODUCER = "imgloss-a1-route-frontier-1"
"""Producer identity carried by every canonical A1 frontier record."""

ROUTE_A_STAGES: tuple[str, ...] = (
    "admission", "camera", "facade_grid", "palette_label", "terminal",
)
"""Route A's frozen checkpoint order (normal-mode single-view facade)."""

ROUTE_B_STAGES: tuple[str, ...] = ("entry", "lattice", "geometry", "output")
"""Route B's frozen checkpoint order (image-derived multiview production)."""

_JSON_PRIMITIVES = (str, bool, int, float)


class FrontierRefusal(ValueError):
    """Raised when a frontier cannot be built from the records it was given.

    The refusal is atomic: a caller never receives a partially built frontier.
    """


class RecordKind(Enum):
    """The metric role of one checkpoint; the kinds are never interchangeable."""

    GEOMETRY = "G"
    OWNERSHIP = "O"
    PAIR_CORRESPONDENCE = "P"
    CANDIDATE = "C"
    EXACT_STATE = "E"
    CAMERA_DIAGNOSTIC = "camera"


class Universe(Enum):
    """The fixed universe a checkpoint's numerator and denominator are drawn from."""

    REFERENCE_CELLS = "frozen_reference_cells"
    FRAME_PIXELS = "frozen_frame_pixels"
    FACE_OBSERVATIONS = "cell_slot_frame_face_observations"
    PIXEL_PAIRS = "frozen_pixel_pair_identities"
    NONE = "none"


class Route(Enum):
    """The image-only entry route a frontier measures."""

    A = "route_a"
    B = "route_b"


# Each frozen plan entry is (stage, kind, universe, record_state_when_reached).
_ROUTE_A_PLAN: tuple[tuple[str, RecordKind, Universe, str], ...] = (
    ("admission", RecordKind.GEOMETRY, Universe.FRAME_PIXELS, "ok"),
    ("camera", RecordKind.CAMERA_DIAGNOSTIC, Universe.NONE, "ok"),
    ("facade_grid", RecordKind.GEOMETRY, Universe.REFERENCE_CELLS, "assembled"),
    ("palette_label", RecordKind.EXACT_STATE, Universe.REFERENCE_CELLS, "ok"),
    ("terminal", RecordKind.EXACT_STATE, Universe.REFERENCE_CELLS, "available"),
)

_ROUTE_B_PLAN: tuple[tuple[str, RecordKind, Universe, str], ...] = (
    ("entry", RecordKind.PAIR_CORRESPONDENCE, Universe.PIXEL_PAIRS, "ok"),
    ("lattice", RecordKind.GEOMETRY, Universe.REFERENCE_CELLS, "assembled"),
    ("geometry", RecordKind.GEOMETRY, Universe.REFERENCE_CELLS, "assembled"),
    ("output", RecordKind.EXACT_STATE, Universe.REFERENCE_CELLS, "decided"),
)


def canonical_json(value: Any) -> str:
    """Canonical, key-sorted, ASCII JSON for one deterministic payload."""
    try:
        return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)
    except FrontierRefusal:
        raise
    except (TypeError, ValueError) as error:
        raise FrontierRefusal("frontier payload is not canonical JSON") from error


def _json_value(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if value != value or value in (float("inf"), float("-inf")):
            raise FrontierRefusal("frontier payload floats must be finite")
        return value
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise FrontierRefusal("frontier payload keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    raise FrontierRefusal("frontier payload is not a deterministic JSON value")


def record_digest(payload: Any) -> str:
    """SHA-256 of one canonical payload, as lowercase hex."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _is_digest(value: Any) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _is_token(value: Any) -> bool:
    """A non-empty single token (no whitespace): a stage or a typed reason."""
    return type(value) is str and bool(value) and not any(
        character.isspace() for character in value)


def _is_text(value: Any) -> bool:
    """Any non-empty string; acceptance states like ``geometry_unavailable``."""
    return type(value) is str and bool(value)


def _checkpoint_payload(value: "FrontierCheckpoint") -> dict:
    kind = value.kind.value if isinstance(value.kind, RecordKind) else value.kind
    universe = (value.universe.value
                if isinstance(value.universe, Universe) else value.universe)
    return {
        "stage": value.stage,
        "kind": kind,
        "universe": universe,
        "reachable": value.reachable,
        "record_state": value.record_state,
        "record_digest": value.record_digest,
        "record_detail": value.record_detail,
        "typed_reason": value.typed_reason,
        "unavailable_reason": value.unavailable_reason,
        "blocked": value.blocked,
        "metric": value.metric,
        "evidence_digest": value.evidence_digest,
    }


def _transition_payload(value: "FrontierTransition") -> dict:
    return {
        "universe": value.universe.value,
        "previous_stage": value.previous_stage,
        "next_stage": value.next_stage,
        "previous_sha256": value.previous_sha256,
        "next_sha256": value.next_sha256,
        "previous": value.previous,
        "gained": value.gained,
        "lost": value.lost,
        "net": value.net,
        "geometry_lost": value.geometry_lost,
        "state_lost": value.state_lost,
    }


def _frontier_payload(value: "ImageOnlyFrontier") -> dict:
    return {
        "schema": value.schema,
        "producer": value.producer,
        "route": value.route.value,
        "checkpoints": [item.to_record() for item in value.checkpoints],
        "transitions": [item.to_record() for item in value.transitions],
        "terminal_policy": (None if value.terminal_policy is None
                            else dict(value.terminal_policy)),
        "terminal_policy_agrees": value.terminal_policy_agrees,
        "terminal_status": value.terminal_status,
        "terminal_reason": value.terminal_reason,
        "terminal_detail": value.terminal_detail,
    }


@dataclass(frozen=True, slots=True)
class FrontierCheckpoint:
    """One route checkpoint and the typed evidence its run carried for it.

    ``reachable`` is the checkpoint's only count-like statement: it is whether the
    run carried the record for this stage, never how many cells were recovered.
    An unreachable checkpoint carries ``metric=None``, no ``record_digest``, and a
    non-empty ``blocked`` token, so a missing record can never be read as a
    measured zero.  A camera checkpoint is a diagnostic: its universe is ``NONE``
    and it must never carry a cell denominator or a transition term.
    """

    stage: str
    kind: RecordKind
    universe: Universe
    reachable: bool
    record_state: str = ""
    record_digest: str = ""
    record_detail: str = ""
    typed_reason: str = ""
    unavailable_reason: str | None = None
    blocked: str = ""
    metric: None = None
    evidence_digest: str = ""
    checkpoint_sha256: str = ""

    def __post_init__(self) -> None:
        if not _is_token(self.stage):
            raise FrontierRefusal("a checkpoint needs a non-empty stage token")
        if not isinstance(self.kind, RecordKind):
            raise FrontierRefusal("a checkpoint kind must be a RecordKind")
        if not isinstance(self.universe, Universe):
            raise FrontierRefusal("a checkpoint universe must be a Universe")
        if type(self.reachable) is not bool:
            raise FrontierRefusal("a checkpoint reachable flag must be a boolean")
        if self.kind is RecordKind.CAMERA_DIAGNOSTIC \
                and self.universe is not Universe.NONE:
            raise FrontierRefusal(
                "a camera diagnostic has no cell denominator and no universe")
        if self.unavailable_reason is not None \
                and type(self.unavailable_reason) is not str:
            raise FrontierRefusal("checkpoint unavailable_reason must be a string or null")
        for name in ("record_state", "record_detail", "typed_reason",
                     "blocked", "evidence_digest"):
            if type(getattr(self, name)) is not str:
                raise FrontierRefusal(f"checkpoint {name} must be a string")
        if self.metric is not None:
            raise FrontierRefusal(
                "A1 carries no numerator or denominator; metric must be null")
        if self.reachable:
            if self.blocked:
                raise FrontierRefusal("a reachable checkpoint cannot carry a blocked token")
            if self.unavailable_reason is not None:
                raise FrontierRefusal(
                    "a reachable checkpoint cannot carry an unavailable reason")
        else:
            if not self.blocked and not self.typed_reason:
                raise FrontierRefusal(
                    "an unreachable checkpoint needs a blocked or typed reason")
            if self.record_digest:
                raise FrontierRefusal(
                    "an unreachable checkpoint cannot carry a record digest")
        if self.record_digest and not _is_digest(self.record_digest):
            raise FrontierRefusal("checkpoint record_digest must be lowercase SHA-256")
        if self.evidence_digest and not _is_digest(self.evidence_digest):
            raise FrontierRefusal("checkpoint evidence_digest must be lowercase SHA-256")
        expected = record_digest(_checkpoint_payload(self))
        if self.checkpoint_sha256 and self.checkpoint_sha256 != expected:
            raise FrontierRefusal("checkpoint digest mismatch")
        if not self.checkpoint_sha256:
            object.__setattr__(self, "checkpoint_sha256", expected)

    @property
    def reason(self) -> str:
        """The checkpoint's typed reason, preferring the blocked token."""
        return self.blocked or self.typed_reason

    def to_record(self) -> dict:
        record = _checkpoint_payload(self)
        record["checkpoint_sha256"] = self.checkpoint_sha256
        return record


@dataclass(frozen=True, slots=True)
class FrontierTransition:
    """A gain/loss step between two comparable checkpoints in one universe.

    A1 has no truth denominator, so every numeric field is ``None``: this type
    fixes the record shape and the digest bindings so A2 can fill the counts in
    without changing the contract.  A1 performs no gain/loss arithmetic and does
    not reconcile or validate numeric transitions; ``geometry_lost``/``state_lost``
    are reserved for A2's cell-level exact-state transitions and are ``None``
    here.  A non-null value in any numeric field (including ``0``) is a typed
    refusal: A1 never carries a count, and its shape can never be mistaken for a
    measured zero.  Camera diagnostics carry no transition at all.
    """

    previous_stage: str
    next_stage: str
    universe: Universe
    previous_sha256: str = ""
    next_sha256: str = ""
    previous: int | None = None
    gained: int | None = None
    lost: int | None = None
    net: int | None = None
    geometry_lost: int | None = None
    state_lost: int | None = None
    transition_sha256: str = ""

    def __post_init__(self) -> None:
        if not _is_token(self.previous_stage) or not _is_token(self.next_stage):
            raise FrontierRefusal("a transition needs two non-empty stage tokens")
        if self.previous_stage == self.next_stage:
            raise FrontierRefusal("a transition needs two distinct stages")
        if not isinstance(self.universe, Universe) or self.universe is Universe.NONE:
            raise FrontierRefusal(
                "a transition needs one comparable non-diagnostic universe")
        # A1 carries no count in any numeric transition field.  ``0`` is as
        # refuseable as any other value: a missing count is typed ``None`` and
        # must never be read as a measured zero.
        for name in ("previous", "gained", "lost", "net",
                     "geometry_lost", "state_lost"):
            if getattr(self, name) is not None:
                raise FrontierRefusal(
                    f"A1 carries no transition counts; {name} must be null")
        if self.previous_sha256 and not _is_digest(self.previous_sha256):
            raise FrontierRefusal("transition previous_sha256 must be lowercase SHA-256")
        if self.next_sha256 and not _is_digest(self.next_sha256):
            raise FrontierRefusal("transition next_sha256 must be lowercase SHA-256")
        expected = record_digest(_transition_payload(self))
        if self.transition_sha256 and self.transition_sha256 != expected:
            raise FrontierRefusal("transition digest mismatch")
        if not self.transition_sha256:
            object.__setattr__(self, "transition_sha256", expected)

    def to_record(self) -> dict:
        record = _transition_payload(self)
        record["transition_sha256"] = self.transition_sha256
        return record


@dataclass(frozen=True, slots=True)
class ImageOnlyFrontier:
    """One route's canonical A1 frontier: checkpoints, transitions and terminal.

    ``terminal_policy`` is the policy the run's own accepted record reports (for
    example the schema-3 envelope's ``stage``/``status``/``reason``); the
    ``terminal_*`` fields are the frontier's own reading.  ``terminal_policy_agrees``
    is the binding check that the two are the same, and no frontier whose policy
    disagrees can be built.
    """

    route: Route
    checkpoints: tuple[FrontierCheckpoint, ...]
    transitions: tuple[FrontierTransition, ...] = ()
    terminal_policy: Mapping[str, Any] | None = None
    terminal_policy_agrees: bool = False
    terminal_status: str = ""
    terminal_reason: str = ""
    terminal_detail: str = ""
    schema: str = FRONTIER_SCHEMA
    producer: str = FRONTIER_PRODUCER
    frontier_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema != FRONTIER_SCHEMA or self.producer != FRONTIER_PRODUCER:
            raise FrontierRefusal("frontier schema or producer is unsupported")
        if not isinstance(self.route, Route):
            raise FrontierRefusal("a frontier route must be a Route")
        if not isinstance(self.checkpoints, tuple) or not self.checkpoints:
            raise FrontierRefusal("a frontier needs at least one checkpoint")
        if any(not isinstance(item, FrontierCheckpoint)
               for item in self.checkpoints):
            raise FrontierRefusal("frontier checkpoints must be checkpoint records")
        stages = tuple(item.stage for item in self.checkpoints)
        if len(set(stages)) != len(stages):
            raise FrontierRefusal("frontier checkpoint stages must be unique")
        expected_stages = ROUTE_A_STAGES if self.route is Route.A else ROUTE_B_STAGES
        if stages != expected_stages[:len(stages)]:
            raise FrontierRefusal(
                "frontier checkpoints must be a canonical route prefix")
        # A frontier is a prefix: after the first unreachable checkpoint every row
        # is unreachable, so no later stage can appear recovered.
        flags = [item.reachable for item in self.checkpoints]
        if any(type(flag) is not bool for flag in flags):
            raise FrontierRefusal("every checkpoint reachable flag must be a boolean")
        if any(flags[index] and not all(flags[:index])
               for index in range(len(flags))):
            raise FrontierRefusal(
                "a reachable checkpoint cannot follow an unreachable one")
        if flags != sorted(flags, reverse=True):
            raise FrontierRefusal(
                "frontier reachability must be a contiguous prefix")
        if self.route is Route.A:
            camera = tuple(item for item in self.checkpoints
                           if item.kind is RecordKind.CAMERA_DIAGNOSTIC)
            if len(camera) > 1 or (camera and camera[0].stage != "camera"):
                raise FrontierRefusal(
                    "route A carries exactly one camera diagnostic")
        else:
            if any(item.kind is RecordKind.CAMERA_DIAGNOSTIC
                   for item in self.checkpoints):
                raise FrontierRefusal("route B carries no camera diagnostic")
        # Transitions join one adjacent, consecutive pair of *reached* checkpoints
        # in the same labelled universe.  A pair that crosses two universes (or
        # touches the camera diagnostic) carries no transition, so the carried
        # transitions are an ordered subset of the adjacent pairs.
        if not isinstance(self.transitions, tuple) or any(
                not isinstance(item, FrontierTransition) for item in self.transitions):
            raise FrontierRefusal("frontier transitions must be transition records")
        if len(self.transitions) > len(self.checkpoints) - 1:
            raise FrontierRefusal("frontier has more transitions than gaps")
        index_of = {item.stage: index for index, item in enumerate(self.checkpoints)}
        seen: set[str] = set()
        last_index = -1
        for transition in self.transitions:
            previous_index = index_of[transition.previous_stage]
            following_index = index_of[transition.next_stage]
            if following_index != previous_index + 1:
                raise FrontierRefusal(
                    "transitions must join adjacent checkpoints in canonical order")
            if following_index <= last_index:
                raise FrontierRefusal("transitions must stay in canonical order")
            last_index = following_index
            if transition.next_stage in seen:
                raise FrontierRefusal("a checkpoint may carry at most one transition")
            seen.add(transition.next_stage)
            previous = self.checkpoints[previous_index]
            following = self.checkpoints[following_index]
            if previous.kind is RecordKind.CAMERA_DIAGNOSTIC \
                    or following.kind is RecordKind.CAMERA_DIAGNOSTIC:
                raise FrontierRefusal(
                    "a camera diagnostic carries no transition term")
            if transition.universe is not previous.universe \
                    or transition.universe is not following.universe:
                raise FrontierRefusal(
                    "a transition may only join one comparable universe")
            if not (previous.reachable and following.reachable):
                raise FrontierRefusal(
                    "a transition needs two comparable reached checkpoints")
            if transition.previous_sha256 != previous.checkpoint_sha256 \
                    or transition.next_sha256 != following.checkpoint_sha256:
                raise FrontierRefusal(
                    "a transition must bind the two checkpoints it joins")
        policy = None if self.terminal_policy is None else {
            str(key): self.terminal_policy[key]
            for key in sorted(self.terminal_policy)}
        object.__setattr__(self, "terminal_policy", policy)
        for name in ("terminal_status", "terminal_reason", "terminal_detail"):
            if type(getattr(self, name)) is not str:
                raise FrontierRefusal(f"frontier {name} must be a string")
        if type(self.terminal_policy_agrees) is not bool:
            raise FrontierRefusal("terminal_policy_agrees must be a boolean")
        expected = record_digest(_frontier_payload(self))
        if self.frontier_sha256 and self.frontier_sha256 != expected:
            raise FrontierRefusal("frontier digest mismatch")
        if not self.frontier_sha256:
            object.__setattr__(self, "frontier_sha256", expected)

    @property
    def reached_stages(self) -> tuple[str, ...]:
        """The canonical stages this run reached, in route order."""
        return tuple(item.stage for item in self.checkpoints if item.reachable)

    @property
    def frontier_stop(self) -> FrontierCheckpoint | None:
        """The first unreachable checkpoint, or ``None`` for a complete prefix."""
        for item in self.checkpoints:
            if not item.reachable:
                return item
        return None

    @property
    def complete(self) -> bool:
        """Whether every checkpoint in this route's canonical plan was reached."""
        return all(item.reachable for item in self.checkpoints)

    def to_record(self) -> dict:
        record = _frontier_payload(self)
        record["frontier_sha256"] = self.frontier_sha256
        return record


# ---------------------------------------------------------------------------
# Typed fail-closed helpers
# ---------------------------------------------------------------------------

def frontier_unavailable(reason: str, *, stage: str = "", detail: str = "") -> dict:
    """One typed fail-closed section for a route that could not be measured.

    Mirrors ``scripts/benchmark_templates._stagewise_unavailable``: a run that
    could not produce a record is ``available: false`` with a typed reason and
    contributes no zero anywhere.
    """
    if not _is_token(reason):
        raise FrontierRefusal("an unavailable frontier needs a typed reason token")
    return {
        "available": False,
        "stage": stage,
        "reason": reason,
        "detail": detail,
        "metric": None,
    }


def first_observable_failure(frontier: ImageOnlyFrontier) -> dict:
    """The run's terminal failure, and separately its first unattempted checkpoint.

    The terminal tuple is the run's own accepted policy reading (for a supplied
    Route-B result, the validated ``stage``/``status``/``reason``; otherwise the
    frontier's own reading).  ``first_unattempted_checkpoint`` is the earliest
    checkpoint whose record was not carried.  The two are deliberately distinct:
    a carried geometry record with status ``unavailable`` is the terminal failure
    while ``output`` is the first unattempted checkpoint.

    This is a **frontier indicator** - where the run stopped - and never a causal
    attribution of the loss.  ``evidence_digest`` is the accepted digest of the
    last reached record, or the empty string when no checkpoint was reached.
    """
    if not isinstance(frontier, ImageOnlyFrontier):
        raise FrontierRefusal("first_observable_failure needs a frontier record")
    stop = frontier.frontier_stop
    policy = frontier.terminal_policy or {}
    evidence = ""
    for item in frontier.checkpoints:
        if item.reachable:
            evidence = item.record_digest or item.evidence_digest \
                or item.checkpoint_sha256
        else:
            break
    unattempted = None
    if stop is not None:
        unattempted = {
            "stage": stop.stage,
            "kind": stop.kind.value,
            "universe": stop.universe.value,
            "blocked": stop.blocked,
            "typed_reason": stop.reason,
        }
    return {
        "route": frontier.route.value,
        "terminal_stage": str(policy.get("stage", "")),
        "terminal_status": frontier.terminal_status,
        "terminal_reason": frontier.terminal_reason,
        "first_unattempted_checkpoint": unattempted,
        "evidence_digest": evidence,
        "metric": None,
    }


# ---------------------------------------------------------------------------
# Canonical-prefix construction (fail-closed, typed, never a zero)
# ---------------------------------------------------------------------------

def _canonical_prefix(plan, carried, *, reached, blocked_reason=""):
    """Build one route's checkpoint prefix from the records it carried.

    A checkpoint is reachable exactly when its own required record was carried
    and every earlier checkpoint was reached.  The first checkpoint that is not
    reached carries ``<parent>_<reason>`` as its ``blocked`` token and a typed
    ``unavailable_reason`` (or ``null`` when the reason belongs to the run's own
    accepted record); every later checkpoint carries the same typed reason with a
    chain token.  No unreachable checkpoint ever carries a digest or a count.
    """
    reachable: list[bool] = []
    previous_reached = True
    for stage, _kind, _universe, _state in plan:
        previous_reached = previous_reached and bool(carried[stage])
        reachable.append(previous_reached)

    checkpoints: list[FrontierCheckpoint] = []
    cause_stage = ""
    cause = ""
    for index, (stage, kind, universe, reached_state) in enumerate(plan):
        if reachable[index]:
            checkpoints.append(reached(stage, kind, universe, reached_state))
            continue
        if cause_stage:
            # An earlier stage already failed, so this stage repeats that same
            # failure attributed to the stage that first failed, never as a zero.
            blocked = f"{cause_stage}_{cause}"
        elif blocked_reason:
            # The run itself reported why this stage failed; later stages repeat
            # that same failure attributed to this stage.
            cause_stage, cause, blocked = stage, blocked_reason, blocked_reason
        else:
            # No record and no reported failure: the stage is simply absent, and
            # every later stage repeats that absence attributed to this stage.
            cause_stage = stage
            cause = blocked = f"{stage}_record_absent"
        checkpoints.append(FrontierCheckpoint(
            stage=stage, kind=kind, universe=universe, reachable=False,
            record_state="unavailable", typed_reason=blocked_reason,
            blocked=blocked, unavailable_reason=blocked_reason or None))
    return checkpoints


# ---------------------------------------------------------------------------
# Route A (intended single-view facade) adapters
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RouteAFrontier:
    """Route A's frontier plus the production records it was derived from.

    The carried records are the solved camera, the facade ``BlockVolume``, the
    palette the run actually used and the decoded export ``BlockVolume`` - all
    solved or derived production outputs, never evaluator truth.  A2 scores their
    geometry and labels through the same adapters under the permitted alignment.
    """

    frontier: ImageOnlyFrontier
    admission_present: bool = False
    camera: Any = None
    facade: Any = None
    palette: tuple[Any, ...] | None = None
    decoded_export: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.frontier, ImageOnlyFrontier):
            raise FrontierRefusal("a route A result needs a frontier record")
        if self.frontier.route is not Route.A:
            raise FrontierRefusal("a route A result needs a route A frontier")
        if type(self.admission_present) is not bool:
            raise FrontierRefusal("admission_present must be a boolean")
        if not self.admission_present:
            if any(item is not None for item in
                   (self.camera, self.facade, self.palette, self.decoded_export)):
                raise FrontierRefusal(
                    "no route A production record can exist without admission")
            return
        if self.camera is None and any(item is not None for item in
                                       (self.facade, self.palette,
                                        self.decoded_export)):
            raise FrontierRefusal(
                "route A geometry needs a solved camera diagnostic")
        if self.facade is None and self.decoded_export is not None:
            raise FrontierRefusal("no export can exist without a facade volume")
        if self.facade is not None and self.palette is None:
            raise FrontierRefusal("a facade grid needs the palette the run used")

    def to_record(self) -> dict:
        return {
            "frontier": self.frontier.to_record(),
            "admission_present": self.admission_present,
            "camera_present": self.camera is not None,
            "facade_block_count": (None if self.facade is None
                                   else len(getattr(self.facade, "blocks", ()))),
            "palette_size": None if self.palette is None else len(self.palette),
            "decoded_block_count": (
                None if self.decoded_export is None
                else len(getattr(self.decoded_export, "blocks", ()))),
        }


def _volume_block_count(volume: Any, field: str) -> int:
    if volume is None:
        return 0
    try:
        blocks = getattr(volume, "blocks")
    except AttributeError as error:
        raise FrontierRefusal(f"the {field} record is not a block volume") from error
    if blocks is None:
        return 0
    try:
        return len(blocks)
    except TypeError as error:
        raise FrontierRefusal(f"the {field} record has no countable blocks") from error


def route_a_frontier(
    *,
    admission=None,
    camera=None,
    facade=None,
    palette=None,
    decoded_export=None,
    camera_solved: bool | None = None,
    blocked_reason: str = "",
) -> RouteAFrontier:
    """Build Route A's frontier from the normal-mode facade's own outputs.

    ``admission`` is the loaded frame (or ``None`` when the image never loaded);
    ``camera`` is the adapted ``CameraModel`` (or ``None`` when the image-only
    solve failed); ``facade`` is the ``reconstruct_facade`` ``BlockVolume``;
    ``palette`` is the palette the run actually used; ``decoded_export`` is the
    decoded ``.litematic`` round trip.  All are in production order.

    ``camera_solved`` records the camera diagnostic's own outcome (``None``
    derives it from whether ``camera`` was supplied).  ``blocked_reason`` is the
    runner's typed token for the first failed prerequisite (for example
    ``camera_unavailable`` or ``no_visible_blocks``).  This is frontier/status
    measurement only: no evaluator truth, reference, mask or alignment may enter,
    and no visible-truth number is produced.
    """
    if blocked_reason and not _is_token(blocked_reason):
        raise FrontierRefusal("a blocked reason must be a non-empty token")
    if camera_solved is None:
        camera_solved = camera is not None
    if type(camera_solved) is not bool:
        raise FrontierRefusal("camera_solved must be a boolean")
    if camera_solved and camera is None:
        raise FrontierRefusal(
            "a solved camera diagnostic must carry its CameraModel")
    if admission is None:
        if any(item is not None
               for item in (camera, facade, palette, decoded_export)):
            raise FrontierRefusal(
                "no route A production record can exist without admission")
    elif not camera_solved and facade is not None:
        raise FrontierRefusal("no facade volume can exist without a solved camera")
    if facade is None and decoded_export is not None:
        raise FrontierRefusal("no export can exist without a facade volume")

    camera_reached = camera is not None
    carried = {
        "admission": admission is not None,
        "camera": camera_reached,
        "facade_grid": bool(camera_reached and facade is not None
                            and _facade_is_usable(facade)),
    }
    carried["palette_label"] = (carried["facade_grid"] and palette is not None
                                and len(tuple(palette)) > 0)
    carried["terminal"] = (carried["palette_label"]
                           and decoded_export is not None)
    checkpoints = _canonical_prefix(
        _ROUTE_A_PLAN, carried,
        reached=lambda stage, kind, universe, state: FrontierCheckpoint(
            stage=stage, kind=kind, universe=universe, reachable=True,
            record_state=state,
            record_digest=_route_a_digest(
                stage, admission, camera, facade, palette, decoded_export),
            record_detail=_route_a_detail(
                stage, admission, facade, palette, decoded_export)),
        blocked_reason=blocked_reason)

    # No accepted invocation record exists for Route A, so the frontier's own
    # terminal reading is the stage the run actually failed at (the first
    # unattempted checkpoint), or the planned terminal when every stage ran.
    first_unattempted = next((item for item in checkpoints
                              if not item.reachable), None)
    terminal = (first_unattempted if first_unattempted is not None
                else checkpoints[-1])
    frontier = ImageOnlyFrontier(
        route=Route.A, checkpoints=tuple(checkpoints),
        transitions=(),
        terminal_policy={
            "route": Route.A.value,
            "stage": terminal.stage,
            "status": terminal.record_state,
            "reason": terminal.reason,
        },
        terminal_policy_agrees=True,
        terminal_status=terminal.record_state,
        terminal_reason=terminal.reason,
        terminal_detail=terminal.record_detail,
    )
    return RouteAFrontier(
        frontier=frontier,
        admission_present=admission is not None,
        camera=camera, facade=facade, palette=palette,
        decoded_export=decoded_export)


def _facade_is_usable(facade: Any) -> bool:
    try:
        return bool(getattr(facade, "blocks"))
    except AttributeError as error:
        raise FrontierRefusal("the facade record is not a block volume") from error


def _route_a_shape(image: Any) -> list[int]:
    """A canonical, JSON-native shape token for the admitted frame."""
    shape = getattr(image, "shape", None)
    if shape is None:
        data = getattr(image, "data", None)
        shape = getattr(data, "shape", None)
    if shape is None:
        return []
    try:
        return [int(value) for value in shape]
    except (TypeError, ValueError):
        return []


def _route_a_digest(stage, admission, camera, facade, palette, decoded_export):
    if stage == "admission":
        return record_digest({"stage": stage,
                              "shape": _route_a_shape(admission)})
    if stage == "camera":
        return record_digest({
            "stage": stage,
            "position": [float(value) for value in
                         getattr(camera, "position", ())],
            "yaw": float(getattr(camera, "yaw_degrees", 0.0)),
            "pitch": float(getattr(camera, "pitch_degrees", 0.0)),
            "roll": float(getattr(camera, "roll_degrees", 0.0)),
            "fov": float(getattr(camera, "fov_degrees", 0.0)),
        })
    if stage == "facade_grid":
        return record_digest({"stage": stage, "volume": _volume_record(facade)})
    if stage == "palette_label":
        return record_digest({
            "stage": stage,
            "palette": [_palette_token(item) for item in palette],
            "volume": _volume_record(facade),
        })
    if decoded_export is None:
        return ""
    return record_digest({"stage": stage,
                          "volume": _volume_record(decoded_export)})


def _volume_record(volume: Any) -> Any:
    """A canonical digest payload for a block volume shaped record."""
    to_dict = getattr(volume, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    blocks = getattr(volume, "blocks", None)
    if blocks is None:
        raise FrontierRefusal("the record is not a block volume")
    payload = []
    for coordinate, observation in sorted(blocks.items()):
        state = getattr(observation, "state", observation)
        name = getattr(state, "name", state)
        payload.append([list(coordinate), str(name)])
    return {"blocks": payload}


def _palette_token(item: Any) -> Any:
    state = getattr(item, "state", None)
    name = getattr(state, "name", None)
    if name is None and isinstance(state, str):
        name = state
    rgb = getattr(item, "rgb", None)
    values = None
    if rgb is not None:
        try:
            values = [int(value) for value in rgb]
        except (TypeError, ValueError):
            values = [str(value) for value in rgb]
    return {"name": str(name), "rgb": values}


def _route_a_detail(stage, admission, facade, palette, decoded_export) -> str:
    if stage == "admission":
        return "frame=" + "x".join(str(value)
                                   for value in _route_a_shape(admission))
    if stage == "camera":
        return ""
    if stage == "facade_grid":
        return f"blocks={_volume_block_count(facade, 'facade')}"
    if stage == "palette_label":
        return f"palette={len(tuple(palette))}"
    return f"blocks={_volume_block_count(decoded_export, 'decoded export')}"


# ---------------------------------------------------------------------------
# Route B (intended multiview production) adapters
# ---------------------------------------------------------------------------

def route_b_frontier(
    *,
    entry=None,
    lattice=None,
    geometry=None,
    output=None,
    result=None,
    blocked_reason: str = "",
) -> ImageOnlyFrontier:
    """Build Route B's frontier from the production chain's own returned records.

    ``entry`` is the in-memory ``ImageDerivedEntryResult``, ``lattice`` the
    ``ImageDerivedLatticeResult``, ``geometry`` the ``ImageDerivedGeometryResult``,
    and ``output`` the ``ImageDerivedOutputResult`` (or ``None`` at any stage the
    run never reached).  ``result`` is the full ``ImageDerivedProductionResult``
    from the same invocation when one is available: its terminal
    ``stage``/``status``/``reason`` becomes the frontier's terminal policy, its
    terminal stage must be the last carried record and must bind that record's
    digest, and a record that contradicts the carried prefix is a typed refusal
    rather than a frontier.

    ``blocked_reason`` is the runner's typed token for an entry that never
    returned (for example ``entry_refused``).  This is frontier/status measurement
    only: no reference, mask, camera manifest or FABRICATED coordinate may enter.
    """
    if blocked_reason and not _is_token(blocked_reason):
        raise FrontierRefusal("a blocked reason must be a non-empty token")
    carried: dict[str, bool] = {
        "entry": entry is not None,
        "lattice": lattice is not None,
        "geometry": geometry is not None,
        "output": output is not None,
    }
    if not carried["entry"] and any(carried[name] for name in
                                    ("lattice", "geometry", "output")):
        raise FrontierRefusal("route B cannot carry a later record without entry")
    if not carried["lattice"] and any(carried[name] for name in
                                      ("geometry", "output")):
        raise FrontierRefusal("route B cannot carry geometry without a lattice")
    if not carried["geometry"] and carried["output"]:
        raise FrontierRefusal("route B cannot carry output without geometry")
    _replay_bindings(carried, entry, lattice, geometry, output)

    records = {"entry": entry, "lattice": lattice, "geometry": geometry,
               "output": output}

    def _reached(stage, kind, universe, reached_state):
        record = records[stage]
        state, reason, detail = _route_b_state(stage, record, reached_state)
        return FrontierCheckpoint(
            stage=stage, kind=kind, universe=universe, reachable=True,
            record_state=state, record_digest=_route_b_digest(stage, record),
            record_detail=detail, typed_reason=reason)

    checkpoints = _canonical_prefix(
        _ROUTE_B_PLAN, carried, reached=_reached,
        blocked_reason=blocked_reason)
    transitions = _route_b_transitions(checkpoints)

    # ``frontier_stop`` is the first checkpoint whose record was not carried (the
    # first unattempted stage).  That is deliberately not the same concept as the
    # run's terminal: a failed geometry record is carried, so ``geometry`` can be
    # the accepted terminal while ``output`` is the first unattempted checkpoint.
    stop = next((item for item in checkpoints if not item.reachable), None)
    if result is None:
        # No accepted invocation record: the frontier reports the stage the run
        # stopped at as its own terminal reading, in production vocabulary.
        terminal = stop if stop is not None else checkpoints[-1]
        terminal_status = _route_b_terminal_status(
            entry, lattice, geometry, output, terminal.stage)
        terminal_reason = terminal.reason
        terminal_detail = terminal.record_detail
        policy = {
            "route": Route.B.value,
            "stage": terminal.stage,
            "status": terminal_status,
            "reason": terminal_reason,
        }
    else:
        # The accepted invocation record is authoritative for the terminal tuple.
        # Its stage must be the last carried record and, when it publishes one,
        # must bind that record's digest.
        terminal_status, terminal_reason, terminal_detail, policy = \
            _route_b_result_terminal(result, checkpoints)
    return ImageOnlyFrontier(
        route=Route.B, checkpoints=tuple(checkpoints), transitions=transitions,
        terminal_policy=policy, terminal_policy_agrees=True,
        terminal_status=terminal_status, terminal_reason=terminal_reason,
        terminal_detail=terminal_detail,
    )


#: The canonical terminal status the production result publishes at each stage.
_ROUTE_B_TERMINAL_STATUS: dict[str, dict[str, str]] = {
    "entry": {"ok": "geometry_unavailable", "unavailable": "geometry_unavailable"},
    "lattice": {"assembled": "geometry_unavailable",
                "unavailable": "geometry_unavailable"},
    "geometry": {"assembled": "geometry_unavailable",
                 "unavailable": "geometry_unavailable"},
    "output": {"promoted": "promoted", "review": "review", "refused": "refused",
               "budget_exhausted": "budget_exhausted",
               "unavailable": "geometry_unavailable"},
}

#: The existing production terminal-status vocabulary, derived from the mapping.
_ROUTE_B_PRODUCTION_STATUSES: frozenset[str] = frozenset(
    status for mapping in _ROUTE_B_TERMINAL_STATUS.values()
    for status in mapping.values())


def _route_b_terminal_status(entry, lattice, geometry, output,
                             terminal_stage: str) -> str:
    """The frontier's terminal status in the production terminal vocabulary.

    Each route-B stage record carries its own stage-local state (for example a
    lattice ``"unavailable"``); the accepted production result instead publishes
    its terminal ``stage``/``status`` pair (for example
    ``geometry``/``geometry_unavailable``).  This is the one translation between
    the two vocabularies and is used only when no accepted record was supplied.
    """
    record = {"entry": entry, "lattice": lattice, "geometry": geometry,
              "output": output}[terminal_stage]
    if record is None:
        return "unavailable"
    state = str(getattr(record, "status", "") or "")
    mapping = _ROUTE_B_TERMINAL_STATUS[terminal_stage]
    return mapping.get(state, "unavailable")


def _route_b_digest(stage: str, record: Any) -> str:
    for name in ("entry_sha256", "result_sha256",
                 "entry_digest", "result_digest", "graph_sha256"):
        value = getattr(record, name, None)
        if _is_digest(value):
            return value
    digest = getattr(record, "to_record", None)
    if callable(digest):
        return record_digest({"stage": stage, "record": digest()})
    raise FrontierRefusal(
        f"the accepted {stage} record carries no replayable digest")


def _route_b_state(stage: str, record: Any, reached_state: str):
    if stage == "entry":
        status = str(getattr(record, "status", "") or "ok")
        reason = str(getattr(record, "reason", ""))
        session = getattr(record, "session", None)
        frames = len(getattr(session, "frames", ())) if session is not None else 0
        return status, reason, f"frames={frames}"
    if stage in ("lattice", "geometry"):
        status = str(getattr(record, "status", "") or "unavailable")
        reason = str(getattr(record, "reason", ""))
        return status, reason, str(getattr(record, "detail", ""))
    status = str(getattr(record, "status", "") or "unavailable")
    reason = str(getattr(record, "reason", ""))
    return status, reason, str(getattr(record, "detail", ""))


def _route_b_transitions(
    checkpoints: list[FrontierCheckpoint],
) -> tuple[FrontierTransition, ...]:
    """Carried gain/loss steps between comparable reached checkpoints.

    A transition exists only between two *reached* checkpoints that share one
    labelled universe.  ``entry`` lives in the pixel-pair ``P`` universe while the
    later stages live in the reference-cell universe, so entry starts no
    transition: joining the two would be a cross-universe subtraction.  Route B
    carries no truth denominator in A1, so every count is typed ``None``.
    """
    transitions: list[FrontierTransition] = []
    for previous, following in zip(checkpoints, checkpoints[1:]):
        if not (previous.reachable and following.reachable):
            break
        if previous.kind is RecordKind.CAMERA_DIAGNOSTIC \
                or following.kind is RecordKind.CAMERA_DIAGNOSTIC:
            continue
        if previous.universe is not following.universe:
            continue
        transitions.append(FrontierTransition(
            previous_stage=previous.stage, next_stage=following.stage,
            universe=previous.universe,
            previous_sha256=previous.checkpoint_sha256,
            next_sha256=following.checkpoint_sha256))
    return tuple(transitions)


def _route_b_result_terminal(result: Any, checkpoints: list[FrontierCheckpoint]):
    """The validated terminal tuple from one accepted invocation record.

    Returns ``(status, reason, detail, policy)``.  The record's terminal stage must
    be a route-B stage and the last carried record, and any stage digest the record
    publishes must equal that record's own replayable digest.  The status is taken
    verbatim from the record and must be one of the existing production statuses.
    """
    stage = str(getattr(result, "stage", ""))
    status = str(getattr(result, "status", ""))
    reason = str(getattr(result, "reason", ""))
    detail = str(getattr(result, "detail", ""))
    if stage not in ROUTE_B_STAGES:
        raise FrontierRefusal(
            "the accepted result terminal stage is not a route B stage")
    reached = [item for item in checkpoints if item.reachable]
    if not reached or reached[-1].stage != stage:
        raise FrontierRefusal(
            "the accepted result terminal stage is not the last carried record")
    bound = getattr(result, f"{stage}_sha256", "")
    if bound and bound != reached[-1].record_digest:
        raise FrontierRefusal(
            "the accepted result does not bind the terminal record digest")
    if status not in _ROUTE_B_PRODUCTION_STATUSES:
        raise FrontierRefusal(
            "the accepted result terminal status is not production vocabulary")
    policy = {
        "schema": str(getattr(result, "schema", "")),
        "producer": str(getattr(result, "producer", "")),
        "stage": stage, "status": status, "reason": reason, "detail": detail,
        "result_sha256": str(getattr(result, "result_sha256", "")),
    }
    return status, reason, detail, policy


def _replay_bindings(carried, entry, lattice, geometry, output) -> None:
    """Prove the carried records bind each other, in order."""
    entry_digest = getattr(entry, "entry_sha256", None) if entry is not None else None
    if lattice is not None:
        if not _is_digest(entry_digest):
            raise FrontierRefusal("the entry record carries no accepted digest")
        session = getattr(entry, "session", None)
        session_digest = getattr(session, "session_sha256", None)
        if getattr(lattice, "session_sha256", None) != session_digest:
            raise FrontierRefusal("the lattice does not bind the live entry session")
    if geometry is not None:
        if getattr(geometry, "lattice_sha256", None) \
                != getattr(lattice, "result_sha256", None):
            raise FrontierRefusal("the geometry does not bind the lattice")
    if output is not None:
        if getattr(output, "geometry_sha256", None) \
                != getattr(geometry, "result_sha256", None):
            raise FrontierRefusal("the output does not bind the geometry")
