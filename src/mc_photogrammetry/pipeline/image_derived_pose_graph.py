"""Bounded M4 relative pose graph composition over accepted connection evidence.

This is the narrow seam that turns the accepted, digest-bound
:class:`~mc_photogrammetry.pipeline.image_derived_connections.ImageDerivedConnectionGraph`
into deterministic, *relative* per-frame gauge poses - and nothing more. It owns
exactly five things:

* a full replay of the source graph before any budget or composition work, so a
  tampered, resealed-in-place or malformed graph is a typed refusal rather than a
  silent input; every solved edge's ``relative_rotation`` is additionally
  re-validated as a finite ``3x3`` proper rotation (orthonormal, determinant
  ``+1``) within an explicit tolerance, because the accepted evidence
  constructor guarantees only shape and finiteness;
* an explicit :class:`PoseGraphBudgets` record (solved-edge budget, cycle-check
  budget, rotation tolerance, direction tolerance) whose digest binds the caps
  the caller declared;
* a deterministic canonical spanning forest per solved connected component,
  anchored at the component's first frame (identity rotation, zero position) and
  propagated along tree edges in canonical order, with the session's isolated
  frames emitted as identity gauge poses;
* two typed cycle refusals - rotation/quadrant cycle inconsistency and
  direction closure inconsistency - each with an explicit tolerance, so a
  contradictory graph is refused whole rather than averaged or partially
  emitted;
* an immutable :class:`ImageDerivedPoseGraph` result carrying immutable
  :class:`GaugeCameraPose` records, the copied session ``unresolved`` facts, a
  hard-frozen ``metric_scale_known=False``, the source ``graph_sha256`` and
  ``session_sha256`` provenance, and a self-computing deterministic digest.

Edge composition convention (bound exactly, verified against the production
matcher)
--------------------------------------------------------------------------
The accepted evidence record documents two quantities per solved edge
``(left, right)``:

* ``relative_rotation`` ``R`` maps frame-``left`` camera coordinates to
  frame-``right`` camera coordinates;
* ``baseline_direction`` ``b`` is the unit left-eye-to-right-eye vector
  ``C_right - C_left`` **in the world frame declared by the left estimate**.

Because one view fixes yaw only modulo 90 degrees, each frame's declared world
frame differs from the shared session frame by an unresolved per-frame yaw
gauge ``k`` in ``Z_4``; ``right_yaw_class`` recovers the relative gauge
``k_right - k_left (mod 4)``. Two baselines recorded in different left-estimate
frames therefore differ by that 90-degree gauge, and ``relative_rotation`` alone
cannot transport between them (the rotation is gauge-invariant). The exact
convention used here is:

* ``gauge[i]`` accumulates ``right_yaw_class`` along the canonical tree path
  from the anchor (``gauge[anchor] = 0``), giving ``k_i - k_anchor (mod 4)``;
* the edge's recorded baseline is transported into the anchor's declared frame
  with ``Ry90(-90 * gauge[edge.left_id])``; traversal signs flip it when the
  edge is walked right-to-left;
* the position propagation is ``position[child] = position[parent] +
  transported_unit_baseline`` (a unit-edge gauge embedding, never metric);
* the rotation propagation is ``rotation[child] = R @ rotation[parent]``,
  anchored at identity.

A closed cycle is rotation-consistent when its composed rotation is the
identity, and direction-consistent when the origin lies in the convex hull of
its signed, transported unit directions - i.e. when positive (unknown) edge
lengths can close the loop. The direction residual is the exact distance from
the origin to that convex hull, computed by the Caratheodory subset enumeration
(no averaging). Even a fully consistent cycle does not close in the unit-edge
position gauge when its true edge lengths differ; that is exactly why the
direction test is scale-free rather than a positional closure.

Hard non-claims: the result carries no metric scale, no position magnitude, no
world coordinate, bound, scene transform, block volume, face observation,
export, CLI, UI or manual metadata. The residual global orientation of the
component anchor's own estimate is not carried by the graph and is never
invented; positions live in the anchor's declared yaw gauge and rotations are
relative to the anchor camera frame. Scale and the session's unresolved facts
are preserved unchanged.
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from mc_photogrammetry.pipeline.image_derived_connections import (
    ImageDerivedConnectionGraph,
)

__all__ = [
    "IMAGE_DERIVED_POSE_GRAPH_SCHEMA",
    "IMAGE_DERIVED_POSE_GRAPH_VERSION",
    "IMAGE_DERIVED_POSE_GRAPH_PRODUCER",
    "POSE_GRAPH_INPUT_INVALID",
    "POSE_GRAPH_SOURCE_INVALID",
    "POSE_GRAPH_SOURCE_REFUSED",
    "POSE_GRAPH_BUDGET_EXHAUSTED",
    "POSE_GRAPH_CYCLE_ROTATION_INCONSISTENT",
    "POSE_GRAPH_CYCLE_DIRECTION_INCONSISTENT",
    "POSE_GRAPH_REFUSAL_REASONS",
    "ImageDerivedPoseGraphError",
    "PoseGraphBudgets",
    "GaugeCameraPose",
    "ImageDerivedPoseGraph",
    "make_pose_graph_budgets",
    "compose_image_derived_pose_graph",
]

IMAGE_DERIVED_POSE_GRAPH_SCHEMA = "mc-photogrammetry-image-derived-pose-graph"
"""Explicit pose-graph result schema name."""

IMAGE_DERIVED_POSE_GRAPH_VERSION = 1
"""Explicit pose-graph result version; an unknown value is refused, never guessed."""

IMAGE_DERIVED_POSE_GRAPH_PRODUCER = "m4-image-derived-pose-graph-1"
"""Producer identity carried by this module."""

POSE_GRAPH_INPUT_INVALID = "pose_graph_input_invalid"
"""Raised reason: the call itself was malformed (wrong type)."""

POSE_GRAPH_SOURCE_INVALID = "pose_graph_source_invalid"
"""Graph refusal: the source graph does not replay its own bindings."""

POSE_GRAPH_SOURCE_REFUSED = "pose_graph_source_refused"
"""Graph refusal: the source graph's status is not ``ok``."""

POSE_GRAPH_BUDGET_EXHAUSTED = "pose_graph_budget_exhausted"
"""Graph refusal: the solved-edge or cycle-check budget was exceeded."""

POSE_GRAPH_CYCLE_ROTATION_INCONSISTENT = "pose_graph_cycle_rotation_inconsistent"
"""Graph refusal: a cycle's composed rotation/quadrant is not the identity."""

POSE_GRAPH_CYCLE_DIRECTION_INCONSISTENT = "pose_graph_cycle_direction_inconsistent"
"""Graph refusal: a cycle's signed unit directions admit no positive closure."""

POSE_GRAPH_REFUSAL_REASONS: tuple[str, ...] = (
    POSE_GRAPH_SOURCE_INVALID,
    POSE_GRAPH_SOURCE_REFUSED,
    POSE_GRAPH_BUDGET_EXHAUSTED,
    POSE_GRAPH_CYCLE_ROTATION_INCONSISTENT,
    POSE_GRAPH_CYCLE_DIRECTION_INCONSISTENT,
)
"""Every typed ``reason`` an :class:`ImageDerivedPoseGraph` may report."""

_TINY = 1e-12
_ROTATION_TOLERANCE = 1e-6
_FEASIBILITY_EPS = 1e-9
_MAX_CARATHEODORY_SIZE = 4


class ImageDerivedPoseGraphError(ValueError):
    """A malformed call.

    ``reason`` is :data:`POSE_GRAPH_INPUT_INVALID`; ``detail`` is a human
    diagnostic that never replaces the typed reason. Data conditions the caller
    can receive as a typed pose-graph refusal are *not* raised.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(reason if not detail else f"{reason}: {detail}")


@dataclass(frozen=True, slots=True)
class PoseGraphBudgets:
    """Explicit limits and tolerances for one composition; both are declared.

    ``max_edges`` is the solved-edge budget: the number of solved evidence edges
    the composer may consume. It is checked against the graph *before* any tree
    or cycle work. ``max_cycle_checks`` is the cycle-check budget: every
    fundamental cycle consumes one check for its rotation/quadrant closure, and
    its direction closure consumes one check per Caratheodory subset examined,
    so the budget bounds the total elementary work as well as the cycle count.
    ``rotation_tolerance_degrees`` is the largest accepted angle between a
    cycle's composed rotation and the identity. ``direction_tolerance`` is the
    largest accepted distance from the origin to the convex hull of a cycle's
    signed transported unit directions. A limit of ``0`` is legal and
    deliberately strict.
    """

    max_edges: int
    max_cycle_checks: int
    rotation_tolerance_degrees: float
    direction_tolerance: float
    budget_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.max_edges) is not int or self.max_edges < 0:
            raise ValueError("max_edges must be a non-negative integer")
        if type(self.max_cycle_checks) is not int or self.max_cycle_checks < 0:
            raise ValueError("max_cycle_checks must be a non-negative integer")
        _require_non_negative_finite(
            self.rotation_tolerance_degrees, "rotation_tolerance_degrees")
        if float(self.rotation_tolerance_degrees) > 180.0:
            raise ValueError("rotation_tolerance_degrees must not exceed 180")
        _require_non_negative_finite(self.direction_tolerance, "direction_tolerance")
        if float(self.direction_tolerance) > 1.0:
            raise ValueError("direction_tolerance must not exceed 1")
        expected = _digest(_budget_payload(self))
        if self.budget_sha256 is None:
            object.__setattr__(self, "budget_sha256", expected)
        elif not _is_sha256(self.budget_sha256) or self.budget_sha256 != expected:
            raise ValueError("pose-graph budget digest mismatch")

    def to_payload(self) -> dict[str, Any]:
        """Canonical budget payload, digest included."""
        payload = _budget_payload(self)
        payload["budget_sha256"] = self.budget_sha256
        return payload


def make_pose_graph_budgets(
    *,
    max_edges: int,
    max_cycle_checks: int,
    rotation_tolerance_degrees: float,
    direction_tolerance: float,
) -> PoseGraphBudgets:
    """Build a self-validating budget from explicit caps (no hidden defaults)."""
    return PoseGraphBudgets(
        max_edges=max_edges,
        max_cycle_checks=max_cycle_checks,
        rotation_tolerance_degrees=rotation_tolerance_degrees,
        direction_tolerance=direction_tolerance,
    )


@dataclass(frozen=True, slots=True)
class GaugeCameraPose:
    """One frame's immutable relative gauge pose.

    ``rotation`` is the gauge-to-camera rotation, anchored at the component's
    first frame (identity) and propagated by the solved relative rotations.
    ``position`` is the unit-edge gauge position in the component anchor's
    declared world-yaw frame: zero at the anchor and advanced by one unit per
    tree edge in that edge's transported baseline direction. ``component_id``
    is the component's canonical index and ``anchor_frame_id`` its first frame.
    ``graph_sha256`` binds the pose to the accepted source graph. No metric
    scale, magnitude or world coordinate is present.
    """

    frame_id: str
    component_id: int
    anchor_frame_id: str
    rotation: tuple[tuple[float, float, float], ...]
    position: tuple[float, float, float]
    graph_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ValueError("pose frame id must be a non-empty string")
        if type(self.component_id) is not int or self.component_id < 0:
            raise ValueError("pose component id must be a non-negative integer")
        if not isinstance(self.anchor_frame_id, str) or not self.anchor_frame_id:
            raise ValueError("pose anchor frame id must be a non-empty string")
        _require_rotation(self.rotation)
        _require_position(self.position)
        if not _is_sha256(self.graph_sha256):
            raise ValueError("pose graph digest must be a lowercase SHA-256 hex string")

    def to_payload(self) -> dict[str, Any]:
        """Canonical pose payload."""
        return _pose_payload(self)


@dataclass(frozen=True, slots=True)
class ImageDerivedPoseGraph:
    """The deterministic relative gauge-pose result for one accepted graph.

    An ``ok`` result carries exactly one :class:`GaugeCameraPose` per source
    frame, in the source's declared frame order, the canonical solved
    components, and the copied session ``unresolved`` facts. A ``refused``
    result carries a typed :data:`POSE_GRAPH_REFUSAL_REASONS` reason and **no**
    poses and **no** components - never a partial result. ``metric_scale_known``
    is hard-frozen ``False``; ``graph_sha256`` and ``session_sha256`` are the
    source provenance and ``pose_graph_sha256`` is this result's own digest.
    """

    schema: str
    version: int
    status: str
    reason: str
    detail: str
    graph_sha256: str
    session_sha256: str
    budgets: PoseGraphBudgets
    frame_ids: tuple[str, ...]
    components: tuple[tuple[str, ...], ...]
    poses: tuple[GaugeCameraPose, ...]
    unresolved: tuple[str, ...]
    metric_scale_known: bool
    pose_graph_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema != IMAGE_DERIVED_POSE_GRAPH_SCHEMA:
            raise ValueError("unsupported pose-graph schema")
        if (type(self.version) is not int
                or self.version != IMAGE_DERIVED_POSE_GRAPH_VERSION):
            raise ValueError("unsupported pose-graph version")
        if self.status not in ("ok", "refused"):
            raise ValueError("pose-graph status must be ok or refused")
        if not isinstance(self.budgets, PoseGraphBudgets):
            raise ValueError("pose-graph budgets must be PoseGraphBudgets")
        if self.metric_scale_known is not False:
            raise ValueError("an image-derived pose graph cannot claim metric scale")
        if not isinstance(self.detail, str):
            raise ValueError("pose-graph refusal detail must be a string")
        if (not isinstance(self.frame_ids, tuple)
                or any(not isinstance(name, str) or not name for name in self.frame_ids)
                or len(set(self.frame_ids)) != len(self.frame_ids)):
            raise ValueError("frame ids must be a tuple of unique non-empty strings")
        if (not isinstance(self.components, tuple)
                or any(not isinstance(component, tuple)
                       or any(not isinstance(name, str) for name in component)
                       for component in self.components)):
            raise ValueError("components must be a tuple of frame-id tuples")
        if (not isinstance(self.unresolved, tuple)
                or any(not isinstance(note, str) for note in self.unresolved)):
            raise ValueError("unresolved notes must be a tuple of strings")
        if (not isinstance(self.poses, tuple)
                or any(not isinstance(pose, GaugeCameraPose) for pose in self.poses)):
            raise ValueError("poses must be a tuple of gauge poses")
        if self.status == "ok":
            if self.reason != "":
                raise ValueError("an ok pose graph cannot carry a refusal reason")
            if not _is_sha256(self.graph_sha256):
                raise ValueError("an ok pose graph needs the source graph digest")
            if not _is_sha256(self.session_sha256):
                raise ValueError("an ok pose graph needs the source session digest")
            if not self.frame_ids:
                raise ValueError("an ok pose graph needs at least one frame")
            if len(self.poses) != len(self.frame_ids):
                raise ValueError("an ok pose graph carries exactly one pose per frame")
            if tuple(pose.frame_id for pose in self.poses) != self.frame_ids:
                raise ValueError("poses are not in canonical frame order")
            for pose in self.poses:
                if pose.graph_sha256 != self.graph_sha256:
                    raise ValueError("a pose does not bind the source graph digest")
            _require_component_consistency(self.frame_ids, self.components, self.poses)
        else:
            if self.reason not in POSE_GRAPH_REFUSAL_REASONS:
                raise ValueError("a refused pose graph needs a typed refusal reason")
            if self.poses != () or self.components != ():
                raise ValueError("a refused pose graph must not carry a partial result")
        expected = _digest(_pose_graph_payload(self))
        if self.pose_graph_sha256 is None:
            object.__setattr__(self, "pose_graph_sha256", expected)
        elif not _is_sha256(self.pose_graph_sha256) or self.pose_graph_sha256 != expected:
            raise ValueError("pose-graph digest mismatch")

    def to_record(self) -> dict[str, Any]:
        """Canonical pose-graph record, digest included."""
        record = _pose_graph_payload(self)
        record["pose_graph_sha256"] = self.pose_graph_sha256
        return record


def compose_image_derived_pose_graph(
    graph: ImageDerivedConnectionGraph,
    *,
    budgets: PoseGraphBudgets,
) -> ImageDerivedPoseGraph:
    """Compose the accepted graph's solved edges into relative gauge poses.

    The source graph is replayed first (schema/version, status, canonical pair
    order, solved components, per-record and graph digests), then the
    solved-edge budget, then a deterministic canonical spanning forest and its
    fundamental-cycle checks. Any inconsistency or budget stop returns a typed
    refusal with no poses and no components. The session's unresolved facts and
    ``scale_known=False`` are copied; no metric scale, translation magnitude,
    world coordinate or scene transform is produced.
    """
    if not isinstance(graph, ImageDerivedConnectionGraph):
        raise ImageDerivedPoseGraphError(
            POSE_GRAPH_INPUT_INVALID, "expected an accepted ImageDerivedConnectionGraph")
    if not isinstance(budgets, PoseGraphBudgets):
        raise ImageDerivedPoseGraphError(
            POSE_GRAPH_INPUT_INVALID, "expected a PoseGraphBudgets")

    frame_ids = _safe_frame_ids(graph)
    graph_digest = graph.graph_sha256 if isinstance(graph.graph_sha256, str) else ""
    session_digest = (
        graph.session_sha256 if isinstance(graph.session_sha256, str) else "")
    unresolved = _safe_unresolved(graph)

    problem = _source_problem(graph)
    if problem is not None:
        reason, detail = problem
        return _refused_pose_graph(
            graph_sha256=graph_digest, session_sha256=session_digest,
            frame_ids=frame_ids, budgets=budgets, reason=reason, detail=detail,
            unresolved=unresolved)

    solved = tuple(record for record in graph.evidence if record.status == "solved")
    if len(solved) > budgets.max_edges:
        return _refused_pose_graph(
            graph_sha256=graph.graph_sha256, session_sha256=graph.session_sha256,
            frame_ids=frame_ids, budgets=budgets, reason=POSE_GRAPH_BUDGET_EXHAUSTED,
            detail=f"solved edge count {len(solved)} exceeds max_edges "
                   f"{budgets.max_edges}",
            unresolved=tuple(graph.unresolved))
    fundamental_cycles = len(solved) - (len(frame_ids) - len(graph.components))
    if fundamental_cycles > budgets.max_cycle_checks:
        return _refused_pose_graph(
            graph_sha256=graph.graph_sha256, session_sha256=graph.session_sha256,
            frame_ids=frame_ids, budgets=budgets, reason=POSE_GRAPH_BUDGET_EXHAUSTED,
            detail=f"fundamental cycle count {fundamental_cycles} exceeds "
                   f"max_cycle_checks {budgets.max_cycle_checks}",
            unresolved=tuple(graph.unresolved))

    try:
        poses = _compose_poses(graph, solved, budgets)
    except _CycleRotationInconsistent as error:
        return _refused_pose_graph(
            graph_sha256=graph.graph_sha256, session_sha256=graph.session_sha256,
            frame_ids=frame_ids, budgets=budgets,
            reason=POSE_GRAPH_CYCLE_ROTATION_INCONSISTENT, detail=str(error),
            unresolved=tuple(graph.unresolved))
    except _CycleDirectionInconsistent as error:
        return _refused_pose_graph(
            graph_sha256=graph.graph_sha256, session_sha256=graph.session_sha256,
            frame_ids=frame_ids, budgets=budgets,
            reason=POSE_GRAPH_CYCLE_DIRECTION_INCONSISTENT, detail=str(error),
            unresolved=tuple(graph.unresolved))
    except _BudgetExceeded:
        return _refused_pose_graph(
            graph_sha256=graph.graph_sha256, session_sha256=graph.session_sha256,
            frame_ids=frame_ids, budgets=budgets, reason=POSE_GRAPH_BUDGET_EXHAUSTED,
            detail=f"cycle checks exceed max_cycle_checks {budgets.max_cycle_checks}",
            unresolved=tuple(graph.unresolved))
    except (TypeError, ValueError) as error:
        # Defensive seal: even though every solved rotation is re-validated as
        # SO(3) in source replay, a propagated rotation that drifts out of the
        # pose constructor's explicit tolerance is still a source-data
        # condition. It must surface as a typed refusal, never as a raw
        # ValueError from pose/record construction.
        return _refused_pose_graph(
            graph_sha256=graph.graph_sha256, session_sha256=graph.session_sha256,
            frame_ids=frame_ids, budgets=budgets, reason=POSE_GRAPH_SOURCE_INVALID,
            detail=f"source graph does not compose to valid gauge poses: {error}",
            unresolved=tuple(graph.unresolved))

    return ImageDerivedPoseGraph(
        schema=IMAGE_DERIVED_POSE_GRAPH_SCHEMA,
        version=IMAGE_DERIVED_POSE_GRAPH_VERSION,
        status="ok",
        reason="",
        detail="",
        graph_sha256=graph.graph_sha256,
        session_sha256=graph.session_sha256,
        budgets=budgets,
        frame_ids=frame_ids,
        components=tuple(graph.components),
        poses=poses,
        unresolved=tuple(graph.unresolved),
        metric_scale_known=False,
    )


# --- Source replay ------------------------------------------------------------

def _source_problem(graph: ImageDerivedConnectionGraph) -> tuple[str, str] | None:
    """Replay every source binding before any budget or composition work.

    The accepted graph and every evidence record are rebuilt through their own
    constructors by :func:`dataclasses.replace`, so their key sets, schema and
    version, status, canonical pair order, replayed components, forbidden
    claims and deterministic digests are re-checked by the module that owns
    them. A graph that does not replay is a typed source refusal.

    The accepted evidence constructor only guarantees that
    ``relative_rotation`` is a finite ``3x3`` matrix, so every *solved* edge's
    rotation is additionally re-validated here as a proper rotation: finite,
    ``3x3``, orthonormal within :data:`_ROTATION_TOLERANCE`, and determinant
    ``+1`` within the same tolerance. This is a data refusal, so it is returned
    before *any* budget is consulted - a graph carrying a malformed
    ``relative_rotation`` such as ``diag(2, 1, 1)`` is
    :data:`POSE_GRAPH_SOURCE_INVALID` even when every budget is zero, never a
    budget stop and never a raw ``ValueError`` from pose construction.
    """
    try:
        dataclasses.replace(graph)
        dataclasses.replace(graph.budgets)
        for record in graph.evidence:
            dataclasses.replace(record)
    except (TypeError, ValueError, KeyError, IndexError, AttributeError) as error:
        return (POSE_GRAPH_SOURCE_INVALID, f"source graph does not replay: {error}")
    if graph.status != "ok":
        return (POSE_GRAPH_SOURCE_REFUSED,
                f"source graph status is {graph.status!r} ({graph.reason!r})")
    for record in graph.evidence:
        if record.session_sha256 != graph.session_sha256:
            return (POSE_GRAPH_SOURCE_INVALID,
                    "an evidence record does not bind the graph session digest")
    for record in graph.evidence:
        if record.status != "solved":
            continue
        problem = _rotation_problem(
            record.relative_rotation,
            f"solved edge {record.left_id}->{record.right_id} relative_rotation")
        if problem is not None:
            return (POSE_GRAPH_SOURCE_INVALID, problem)
    return None


# --- Composition --------------------------------------------------------------

class _BudgetExceeded(Exception):
    """Internal control flow: the cycle-check budget was exhausted."""


class _CycleRotationInconsistent(Exception):
    """Internal control flow: a cycle's composed rotation is not the identity."""


class _CycleDirectionInconsistent(Exception):
    """Internal control flow: a cycle's directions admit no positive closure."""


class _CheckBudget:
    """A deterministic counter for the elementary cycle-check budget."""

    __slots__ = ("remaining",)

    def __init__(self, limit: int) -> None:
        self.remaining = int(limit)

    def charge(self) -> None:
        if self.remaining <= 0:
            raise _BudgetExceeded()
        self.remaining -= 1


@dataclass(frozen=True, slots=True)
class _Tree:
    """One component's canonical spanning tree and its gauge data."""

    component_id: int
    anchor: str
    parent: dict[str, str | None]
    parent_edge: dict[str, Any]
    gauge: dict[str, int]
    rotation: dict[str, np.ndarray]
    position: dict[str, np.ndarray]
    tree_edges: frozenset[tuple[str, str]]


def _compose_poses(
    graph: ImageDerivedConnectionGraph,
    solved: tuple[Any, ...],
    budgets: PoseGraphBudgets,
) -> tuple[GaugeCameraPose, ...]:
    frame_index = {frame_id: index for index, frame_id in enumerate(graph.frame_ids)}
    component_of: dict[str, int] = {}
    for component_id, component in enumerate(graph.components):
        for frame_id in component:
            component_of[frame_id] = component_id

    adjacency: dict[str, list[Any]] = {frame_id: [] for frame_id in graph.frame_ids}
    for record in solved:
        adjacency[record.left_id].append(record)
        adjacency[record.right_id].append(record)
    for frame_id in graph.frame_ids:
        adjacency[frame_id].sort(key=lambda record: frame_index[
            record.right_id if record.left_id == frame_id else record.left_id])

    records_by_component: list[list[Any]] = [[] for _ in graph.components]
    for record in solved:
        records_by_component[component_of[record.left_id]].append(record)

    checks = _CheckBudget(budgets.max_cycle_checks)
    rotations: dict[str, np.ndarray] = {}
    positions: dict[str, np.ndarray] = {}
    for component_id, component in enumerate(graph.components):
        tree = _grow_tree(component_id, component, adjacency)
        for frame_id in component:
            rotations[frame_id] = tree.rotation[frame_id]
            positions[frame_id] = tree.position[frame_id]
        for record in records_by_component[component_id]:
            if (record.left_id, record.right_id) in tree.tree_edges:
                continue
            checks.charge()
            _check_cycle(record, tree, budgets, checks)

    graph_digest = graph.graph_sha256
    return tuple(
        GaugeCameraPose(
            frame_id=frame_id,
            component_id=component_of[frame_id],
            anchor_frame_id=graph.components[component_of[frame_id]][0],
            rotation=_rotation_tuple(rotations[frame_id]),
            position=_position_tuple(positions[frame_id]),
            graph_sha256=graph_digest,
        )
        for frame_id in graph.frame_ids
    )


def _grow_tree(
    component_id: int, component: tuple[str, ...], adjacency: dict[str, list[Any]],
) -> _Tree:
    """Breadth-first canonical spanning tree, anchored at the component's first frame."""
    anchor = component[0]
    parent: dict[str, str | None] = {anchor: None}
    parent_edge: dict[str, Any] = {anchor: None}
    gauge: dict[str, int] = {anchor: 0}
    rotation: dict[str, np.ndarray] = {anchor: np.eye(3)}
    position: dict[str, np.ndarray] = {anchor: np.zeros(3)}
    tree_edges: set[tuple[str, str]] = set()
    queue = deque([anchor])
    while queue:
        node = queue.popleft()
        for record in adjacency[node]:
            left, right = record.left_id, record.right_id
            child = right if left == node else left
            if child in parent:
                continue
            if left == node:
                step_class = int(record.right_yaw_class)
            else:
                step_class = (-int(record.right_yaw_class)) % 4
            parent[child] = node
            parent_edge[child] = record
            gauge[child] = (gauge[node] + step_class) % 4
            tree_edges.add((left, right))
            edge_rotation = np.asarray(record.relative_rotation, dtype=np.float64)
            rotation[child] = (edge_rotation @ rotation[node] if left == node
                               else edge_rotation.T @ rotation[node])
            step = _transported_direction(record, gauge, left)
            if left != node:
                step = -step
            position[child] = position[node] + step
            queue.append(child)
    return _Tree(
        component_id=component_id,
        anchor=anchor,
        parent=parent,
        parent_edge=parent_edge,
        gauge=gauge,
        rotation=rotation,
        position=position,
        tree_edges=frozenset(tree_edges),
    )


def _check_cycle(
    chord: Any, tree: _Tree, budgets: PoseGraphBudgets, checks: _CheckBudget,
) -> None:
    """Refuse the whole composition when one fundamental cycle contradicts itself.

    The cycle is walked ``left -> ... -> right`` along the canonical tree and
    then back along the chord. Its composed rotation must be the identity within
    ``rotation_tolerance_degrees`` and its quadrant cycle must close modulo 4;
    the origin must lie in the convex hull of its signed transported unit
    directions within ``direction_tolerance``.
    """
    left, right = chord.left_id, chord.right_id
    if (tree.gauge[right] - tree.gauge[left]) % 4 != int(chord.right_yaw_class):
        raise _CycleRotationInconsistent(
            f"cycle {left}->{right} quadrant does not close mod 4")
    loop_rotation = np.eye(3)
    directions: list[np.ndarray] = []
    for edge, source in _tree_path(left, right, tree):
        loop_rotation = _edge_rotation(edge, source) @ loop_rotation
        directions.append(_edge_direction(edge, source, tree.gauge))
    loop_rotation = _edge_rotation(chord, right) @ loop_rotation
    directions.append(_edge_direction(chord, right, tree.gauge))
    angle = _rotation_angle_degrees(loop_rotation)
    if angle > budgets.rotation_tolerance_degrees:
        raise _CycleRotationInconsistent(
            f"cycle through {left}->{right} rotation is {angle:.6f} degrees from identity")
    residual = _caratheodory_residual(np.asarray(directions), checks)
    if residual > budgets.direction_tolerance:
        raise _CycleDirectionInconsistent(
            f"cycle through {left}->{right} direction hull distance is "
            f"{residual:.6f} > {budgets.direction_tolerance}")


def _tree_path(left: str, right: str, tree: _Tree) -> list[tuple[Any, str]]:
    """Edges and traversal sources for the canonical tree walk ``left -> right``."""
    left_chain: list[tuple[str, Any]] = []
    node: str | None = left
    while node is not None:
        left_chain.append((node, tree.parent_edge[node]))
        node = tree.parent[node]
    right_chain: list[tuple[str, Any]] = []
    node = right
    while node is not None:
        right_chain.append((node, tree.parent_edge[node]))
        node = tree.parent[node]
    right_ancestors = {frame_id for frame_id, _ in right_chain}
    lca = next(frame_id for frame_id, _ in left_chain if frame_id in right_ancestors)
    path: list[tuple[Any, str]] = []
    for frame_id, edge in left_chain:
        if frame_id == lca:
            break
        path.append((edge, frame_id))
    descent: list[tuple[Any, str]] = []
    for frame_id, edge in right_chain:
        if frame_id == lca:
            break
        descent.append((edge, tree.parent[frame_id]))
    path.extend(reversed(descent))
    return path


def _transported_direction(record: Any, gauge: dict[str, int], left: str) -> np.ndarray:
    return _ry90(-gauge[left]) @ np.asarray(record.baseline_direction, dtype=np.float64)


def _edge_direction(record: Any, source: str, gauge: dict[str, int]) -> np.ndarray:
    """Unit direction from ``source`` to the record's other frame, anchor gauge."""
    transported = _transported_direction(record, gauge, record.left_id)
    return transported if source == record.left_id else -transported


def _edge_rotation(record: Any, source: str) -> np.ndarray:
    """Rotation from ``source`` camera coordinates to the other frame's."""
    rotation = np.asarray(record.relative_rotation, dtype=np.float64)
    return rotation if source == record.left_id else rotation.T


def _caratheodory_residual(directions: np.ndarray, checks: _CheckBudget) -> float:
    """Exact distance from the origin to the directions' convex hull.

    Every subset of two, three or four unit directions is a candidate face; by
    Caratheodory the origin lies in the hull of at most four of them. Each
    candidate subset consumes one unit of the cycle-check budget. The minimum
    over feasible subsets is the exact hull distance (``0`` means the cycle's
    directions can close with strictly positive lengths). No averaging is used.
    """
    count = int(directions.shape[0])
    best: float | None = None
    for size in range(2, min(_MAX_CARATHEODORY_SIZE, count) + 1):
        for subset in itertools.combinations(range(count), size):
            checks.charge()
            residual = _simplex_residual(directions[list(subset)])
            if residual is not None and (best is None or residual < best):
                best = residual
    return 1.0 if best is None else float(best)


def _simplex_residual(points: np.ndarray) -> float | None:
    """Minimum ``‖sum(x_k p_k)‖`` over that subset's simplex, or ``None``.

    The KKT system ``2Gx - 1*lambda = 0``, ``1ᵀx = 1`` gives the affine-hull
    optimum; it is the simplex optimum exactly when every weight is
    non-negative. Otherwise the optimum lies on an enumerated sub-face.
    """
    size = int(points.shape[0])
    gram = points @ points.T
    system = np.zeros((size + 1, size + 1), dtype=np.float64)
    system[:size, :size] = 2.0 * gram
    system[:size, size] = -1.0
    system[size, :size] = 1.0
    right = np.zeros(size + 1, dtype=np.float64)
    right[size] = 1.0
    try:
        solution = np.linalg.solve(system, right)
    except np.linalg.LinAlgError:
        solution, *_ = np.linalg.lstsq(system, right, rcond=None)
    weights = np.asarray(solution[:size], dtype=np.float64)
    if not np.all(np.isfinite(weights)) or np.any(weights < -_FEASIBILITY_EPS):
        return None
    weights = np.clip(weights, 0.0, None)
    total = float(np.sum(weights))
    if total <= _TINY:
        return None
    weights = weights / total
    return float(np.linalg.norm(points.T @ weights))


# --- Gauge helpers ------------------------------------------------------------

def _ry90(class_index: int) -> np.ndarray:
    """``Ry(90 * class_index)``: the exact 90-degree yaw gauge relabelling."""
    angle = math.radians(90.0 * (int(class_index) % 4))
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)),
        dtype=np.float64,
    )


def _rotation_angle_degrees(rotation: np.ndarray) -> float:
    cosine = float(np.clip((float(np.trace(rotation)) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _rotation_tuple(value: Any) -> tuple[tuple[float, float, float], ...]:
    array = np.asarray(value, dtype=np.float64)
    return tuple(tuple(float(entry) for entry in row) for row in array)


def _position_tuple(value: Any) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    return (float(array[0]), float(array[1]), float(array[2]))


# --- Refusals, safe access and validation -------------------------------------

def _refused_pose_graph(
    *,
    graph_sha256: str,
    session_sha256: str,
    frame_ids: tuple[str, ...],
    budgets: PoseGraphBudgets,
    reason: str,
    detail: str,
    unresolved: tuple[str, ...],
) -> ImageDerivedPoseGraph:
    return ImageDerivedPoseGraph(
        schema=IMAGE_DERIVED_POSE_GRAPH_SCHEMA,
        version=IMAGE_DERIVED_POSE_GRAPH_VERSION,
        status="refused",
        reason=reason,
        detail=detail,
        graph_sha256=graph_sha256,
        session_sha256=session_sha256,
        budgets=budgets,
        frame_ids=frame_ids,
        components=(),
        poses=(),
        unresolved=unresolved,
        metric_scale_known=False,
    )


def _safe_frame_ids(graph: ImageDerivedConnectionGraph) -> tuple[str, ...]:
    frame_ids = getattr(graph, "frame_ids", None)
    if (isinstance(frame_ids, tuple)
            and all(isinstance(name, str) and name for name in frame_ids)):
        return frame_ids
    return ()


def _safe_unresolved(graph: ImageDerivedConnectionGraph) -> tuple[str, ...]:
    value = getattr(graph, "unresolved", ())
    if isinstance(value, tuple) and all(isinstance(note, str) for note in value):
        return value
    return ()


def _require_component_consistency(
    frame_ids: tuple[str, ...],
    components: tuple[tuple[str, ...], ...],
    poses: tuple[GaugeCameraPose, ...],
) -> None:
    grouped: dict[int, list[str]] = {}
    for pose in poses:
        grouped.setdefault(pose.component_id, []).append(pose.frame_id)
    if tuple(tuple(members) for _key, members in sorted(grouped.items())) != components:
        raise ValueError("poses do not replay the declared components")
    if len(components) != len(grouped):
        raise ValueError("poses do not cover every declared component")
    for component_id, component in enumerate(components):
        if not component:
            raise ValueError("a component must name at least one frame")
        anchor = component[0]
        for frame_id in component:
            if frame_id not in frame_ids:
                raise ValueError("a component names an unknown frame")
        anchor_pose = next(pose for pose in poses if pose.frame_id == anchor)
        if anchor_pose.component_id != component_id:
            raise ValueError("a component anchor has the wrong component id")
        if anchor_pose.anchor_frame_id != anchor:
            raise ValueError("a component anchor does not name itself")
        if anchor_pose.rotation != _IDENTITY_ROTATION:
            raise ValueError("a component anchor must sit at identity rotation")
        if anchor_pose.position != (0.0, 0.0, 0.0):
            raise ValueError("a component anchor must sit at zero position")
    for pose in poses:
        if pose.anchor_frame_id != components[pose.component_id][0]:
            raise ValueError("a pose names the wrong component anchor")


_IDENTITY_ROTATION = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _rotation_problem(value: Any, label: str) -> str | None:
    """Return a typed diagnostic when ``value`` is not a proper 3D rotation.

    The exact bound is the module's explicit :data:`_ROTATION_TOLERANCE`: the
    matrix must be finite and ``3x3``, ``AᵀA`` must be the identity within that
    tolerance, and the determinant must be ``+1`` within the same tolerance (so
    a reflection is rejected as well). ``None`` means the value is a valid
    ``SO(3)`` element.

    The orthonormality comparison is an *absolute* entry-wise bound: ``rtol`` is
    forced to ``0.0`` so no NumPy default relative slack (``1e-5``) can widen the
    declared ``1e-6`` tolerance. Determinant-preserving defects just above
    ``1e-6`` must therefore refuse, exactly as documented.
    """
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return f"{label} must be a finite 3x3 matrix"
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        return f"{label} must be a finite 3x3 matrix"
    if not np.allclose(
            array.T @ array, np.eye(3), rtol=0.0, atol=_ROTATION_TOLERANCE):
        return f"{label} must be orthonormal"
    if abs(float(np.linalg.det(array)) - 1.0) > _ROTATION_TOLERANCE:
        return f"{label} must have determinant +1"
    return None


def _require_rotation(value: Any) -> None:
    problem = _rotation_problem(value, "pose rotation")
    if problem is not None:
        raise ValueError(problem)


def _require_position(value: Any) -> None:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError("pose position must be three finite numbers")
    for entry in value:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise ValueError("pose position must be three finite numbers")
        if not math.isfinite(float(entry)):
            raise ValueError("pose position must be three finite numbers")


def _require_non_negative_finite(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number")


# --- Canonical payloads and digests -------------------------------------------

def _budget_payload(budgets: PoseGraphBudgets) -> dict[str, Any]:
    return {
        "max_edges": int(budgets.max_edges),
        "max_cycle_checks": int(budgets.max_cycle_checks),
        "rotation_tolerance_degrees": float(budgets.rotation_tolerance_degrees),
        "direction_tolerance": float(budgets.direction_tolerance),
    }


def _pose_payload(pose: GaugeCameraPose) -> dict[str, Any]:
    return {
        "frame_id": pose.frame_id,
        "component_id": pose.component_id,
        "anchor_frame_id": pose.anchor_frame_id,
        "rotation": [list(row) for row in pose.rotation],
        "position": list(pose.position),
        "graph_sha256": pose.graph_sha256,
    }


def _pose_graph_payload(result: ImageDerivedPoseGraph) -> dict[str, Any]:
    return {
        "schema": result.schema,
        "version": result.version,
        "status": result.status,
        "reason": result.reason,
        "detail": result.detail,
        "graph_sha256": result.graph_sha256,
        "session_sha256": result.session_sha256,
        "budgets": _budget_payload(result.budgets),
        "frame_ids": list(result.frame_ids),
        "components": [list(component) for component in result.components],
        "poses": [_pose_payload(pose) for pose in result.poses],
        "unresolved": list(result.unresolved),
        "metric_scale_known": bool(result.metric_scale_known),
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))
