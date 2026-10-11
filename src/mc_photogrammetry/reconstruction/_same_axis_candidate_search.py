"""Isolated, I/O-free exhaustive candidate experiment; never a geometry selector.

The caller supplies an already authenticated image-derived connection graph and
trusted session/graph pins. Digest replay checks binding and tampering, not image
authenticity: that remains the upstream image-session validator's responsibility.
All feasible complete edge subsets and quadrant assignments survive, regardless
of correlation score. A successful search still authorizes no reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from itertools import product
import hashlib
import json
import math
import sys

import numpy as np

from mc_photogrammetry.pipeline.image_derived_connections import (
    ConnectionBudgets, ImageDerivedConnectionGraph, RelativeCameraEvidence,
    YawClassDiagnostic,
)
from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction.lattice_planes import (
    PlaneReading, reading_order_key,
)


@dataclass(frozen=True, slots=True)
class SearchCaps:
    candidate_identities: int = 64
    search_nodes: int = 4096
    solver_calls: int = 512
    retained_outcomes: int = 512
    diagnostic_bytes: int = 2 * 1024 * 1024
    readings: int = 128
    frames: int = 32


HARD_CAPS = SearchCaps()


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    identities: tuple[tuple, ...]
    frames: tuple[str, ...]
    quadrants: tuple[tuple[str, int], ...]
    translations: tuple[tuple[str, tuple[float, float, float]], ...]
    axis_residuals: tuple[tuple[int, float], ...]
    axis_resultants: tuple[tuple[int, float], ...]
    blocks_per_period: tuple[tuple[tuple[str, int], float], ...]
    solver_reason: str
    scale_known: bool
    alternatives: tuple[plane_graph.ComponentAlternative, ...]


@dataclass(frozen=True, slots=True)
class SearchResult:
    status: str
    reason: str
    outcomes: tuple[CandidateOutcome, ...] = ()
    candidates: tuple[plane_graph.PlaneMatch, ...] = ()
    search_nodes: int = 0
    solver_calls: int = 0
    diagnostic_bytes: int = 0
    reconstruction_authorized: bool = False
    universe_sha256: str = ""


class _Refusal(Exception):
    pass


def representation_bytes(value: object, limit: int = HARD_CAPS.diagnostic_bytes) -> int:
    """Allocated resident size of the referenced Python representation (CPython).

    Unique objects count once. This is neither serialized length nor a bound on
    process peak/RSS, solver temporaries, or this measurement's traversal state.
    Oversize objects stop traversal before their children are expanded.
    """
    seen: set[int] = set()
    total = 0

    stack = [iter((value,))]
    while stack:
        try:
            item = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        if id(item) in seen:
            continue
        seen.add(id(item))
        total += sys.getsizeof(item)
        if total > limit:
            raise _Refusal("diagnostic_bytes_cap")
        if is_dataclass(item) and not isinstance(item, type):
            stack.append(iter(tuple(getattr(item, field.name) for field in fields(item))))
        elif isinstance(item, (tuple, list)):
            stack.append(iter(item))
        elif isinstance(item, dict):
            stack.append((element for pair in item.items() for element in pair))
    return total


def candidate_identity(match: plane_graph.PlaneMatch) -> tuple:
    """Complete indexed geometry identity; scored metrics are payload, not rank.

    Identical identity with different metrics is a conflict, not a new variant.
    The absent shift has its own sentinel so identity sorting stays orderable.
    """
    return (match.left, match.right, match.left_plane, match.right_plane,
            match.left_axis, match.right_axis, match.left_b, match.right_b,
            match.quadrant_of_right, (0,) if match.cell_shift_rc is None
            else (1, *match.cell_shift_rc))


CANDIDATE_UNIVERSE_BASIS = "mc-photogrammetry-same-axis-candidate-universe-v2"
"""Basis tag for the bound kept+dropped candidate-universe digest.

``v2`` binds the full canonical endpoint reading identity (frame/index,
``reading_order_key`` including ``side``, and the selected ``b``/distance). A
``v1`` digest that omitted ``side`` is therefore never replay-comparable.
"""


def _is_sha256(value: object) -> bool:
    """Strict lowercase-hex SHA-256 pin check; non-strings and casing refuse."""
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


@dataclass(frozen=True, slots=True)
class BoundCandidateUniverse:
    """Immutable, canonically ordered kept+dropped candidate universe.

    ``candidates`` and ``dispositions`` are parallel and sorted by
    :func:`candidate_identity`, so a permuted raw observation binds identically.
    ``universe_sha256`` covers every candidate identity, scored payload and
    disposition, both resolved canonical reading endpoints, and the trusted
    session/graph pins. It is a binding digest, never an authority to reconstruct.
    """

    frames: tuple[str, ...]
    candidates: tuple[plane_graph.PlaneMatch, ...]
    dispositions: tuple[str, ...]
    kept_count: int
    dropped_count: int
    session_sha256: str
    graph_sha256: str
    universe_sha256: str


@dataclass(frozen=True, slots=True)
class CandidateUniverseBinding:
    """Typed adapter verdict: one bound universe or one atomic refusal."""

    status: str
    reason: str = ""
    universe: BoundCandidateUniverse | None = None


def _json_identity(match: plane_graph.PlaneMatch) -> list:
    return [list(value) if isinstance(value, tuple) else value
            for value in candidate_identity(match)]


def _json_reading_key(value: object) -> list | str:
    """JSON-safe projection of :func:`lattice_planes.reading_order_key` output.

    The canonical index already proved the key finite and orderable; this only
    makes it serializable and re-checks finiteness, so a malformed or non-finite
    value refuses instead of aborting hashing.
    """
    if type(value) is tuple:
        return [_json_reading_key(item) for item in value]
    if type(value) is str:
        return value
    if type(value) is bool or isinstance(value, np.bool_):
        raise _Refusal("invalid_reading_identity")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise _Refusal("invalid_reading_identity")
        return numeric
    raise _Refusal("invalid_reading_identity")


def _reading_identity_payload(
    reading: PlaneReading,
    index: int,
    selected_b: float,
) -> list:
    """Full canonical reading identity for one bound solver endpoint.

    ``reading_order_key`` is the exact content-only key the matcher and solver use
    to address and order a reading; it carries ``side``, ``axis``, every retained
    distance alternative and the region tie-break. The resolved frame/index and
    the selected ``b`` plus its unique distance are appended.
    """
    distance = reading.distance_for(selected_b)
    if distance is None or not math.isfinite(float(distance)):
        raise _Refusal("invalid_reading_identity")
    return [
        reading.frame, int(index), _json_reading_key(reading_order_key(reading)),
        float(selected_b), float(distance),
    ]


def _candidate_universe_payload(
    ordered: tuple[tuple[plane_graph.PlaneMatch, str], ...],
    index: dict[tuple[str, int], PlaneReading],
    frames: tuple[str, ...],
    session_sha256: str,
    graph_sha256: str,
) -> dict:
    """Deterministic, permutation-invariant digest payload for one universe.

    Each endpoint binds its full canonical reading identity, not just the frame,
    plane index and selected distance. Solver audit: ``solve_component`` and
    ``_evaluate_assignment`` consume only frame/index addressing, ``axis``,
    ``side``, ``alternatives`` (through ``distance_for``) and the selected ``b``.
    All of those are covered by ``reading_order_key`` plus this row, so no
    solver-relevant reading field is omitted. Fields outside the order key
    (rectification, coverage, region/phase evidence content) never reach solver
    geometry.
    """
    rows = []
    for match, disposition in ordered:
        left = index[(match.left, match.left_plane)]
        right = index[(match.right, match.right_plane)]
        rows.append([
            _json_identity(match), disposition,
            _reading_identity_payload(left, match.left_plane, match.left_b),
            _reading_identity_payload(right, match.right_plane, match.right_b),
            float(match.psr), float(match.peak), float(match.second_peak),
            float(match.score), int(match.overlap_cells),
        ])
    return {
        "basis": CANDIDATE_UNIVERSE_BASIS,
        "frames": list(frames),
        "session_sha256": session_sha256,
        "graph_sha256": graph_sha256,
        "rows": rows,
    }


def bind_same_axis_candidate_universe(
    readings: tuple[PlaneReading, ...],
    observation: plane_graph.SameAxisCandidateObservation,
    *,
    connection_graph: ImageDerivedConnectionGraph,
    trusted_session_sha256: str,
    trusted_graph_sha256: str,
    trusted_universe_sha256: str | None = None,
    caps: SearchCaps = HARD_CAPS,
) -> CandidateUniverseBinding:
    """Bind one raw matcher observation to canonical readings and trusted pins.

    Every admitted candidate is forwarded regardless of its keep/drop
    disposition: the local same-axis verdict is bound evidence, not a selector.
    Validation order and contracts match :func:`search_same_axis_candidates`;
    malformed, duplicate, conflicting, over-cap or replay-mismatched bindings
    refuse atomically with no universe. ``trusted_universe_sha256`` is an optional
    replay pin produced by a prior bind; reordering the raw observation cannot
    change the canonical digest, but any content tampering can.
    """
    try:
        if type(caps) is not SearchCaps:
            raise _Refusal("invalid_caps")
        for field in fields(caps):
            value = getattr(caps, field.name)
            if type(value) is not int or not 0 < value <= getattr(HARD_CAPS, field.name):
                raise _Refusal("invalid_caps")
        if type(readings) is not tuple:
            raise _Refusal("invalid_input")
        if type(observation) is not plane_graph.SameAxisCandidateObservation:
            raise _Refusal("invalid_observation")
        if not _is_sha256(trusted_session_sha256) or not _is_sha256(trusted_graph_sha256):
            raise _Refusal("invalid_pins")
        if trusted_universe_sha256 is not None and not _is_sha256(trusted_universe_sha256):
            raise _Refusal("invalid_pins")
        candidates = observation.candidates
        dispositions = observation.dispositions
        if type(candidates) is not tuple or type(dispositions) is not tuple:
            raise _Refusal("invalid_observation")
        if len(candidates) != len(dispositions):
            raise _Refusal("unpaired_observation")
        if len(candidates) > caps.candidate_identities:
            raise _Refusal("candidate_identities_cap")
        if len(readings) > caps.readings:
            raise _Refusal("readings_cap")
        # Measure before canonical indexing/digest replay (which can allocate).
        representation_bytes((readings, observation, connection_graph), caps.diagnostic_bytes)
        index, _, frames = plane_graph._canonical_reading_index(readings)
        if len(frames) > caps.frames:
            raise _Refusal("frames_cap")
        # The same immutable session/graph contracts as the solver-input path.
        _constraints(connection_graph, frames, trusted_session_sha256,
                     trusted_graph_sha256)
        ordered_rows = []
        identities = {}
        for match, disposition in zip(candidates, dispositions):
            plane_graph._validate_indexed_match(match, index, require_selected_b=True)
            plane_graph.PlaneMatch(**{f.name: getattr(match, f.name)
                                     for f in fields(plane_graph.PlaneMatch)})
            if disposition not in (plane_graph.SAME_AXIS_DISPOSITION_KEPT,
                                   plane_graph.SAME_AXIS_DISPOSITION_DROPPED):
                raise _Refusal("invalid_disposition")
            if (match.left >= match.right
                    or type(match.quadrant_of_right) is not int
                    or match.quadrant_of_right not in (0, 1, 2, 3)
                    or plane_graph.axis_transform(match.right_axis, match.quadrant_of_right)[0]
                    != match.left_axis
                    or match.integer_residual is not None
                    or type(match.overlap_cells) is not int or match.overlap_cells < 0):
                raise _Refusal("invalid_candidate")
            for numeric in (match.psr, match.peak, match.second_peak, match.score,
                            match.left_b, match.right_b):
                if type(numeric) not in (float, int) or not math.isfinite(numeric):
                    raise _Refusal("invalid_candidate")
            key = candidate_identity(match)
            if key in identities:
                raise _Refusal("duplicate_identity" if identities[key] == match
                               else "conflicting_identity")
            identities[key] = match
            ordered_rows.append((match, disposition))
        ordered = tuple(sorted(ordered_rows, key=lambda row: candidate_identity(row[0])))
        payload = _candidate_universe_payload(
            ordered, index, frames, trusted_session_sha256, trusted_graph_sha256)
        digest = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False).encode("utf-8")).hexdigest()
        if trusted_universe_sha256 is not None and trusted_universe_sha256 != digest:
            raise _Refusal("binding_mismatch")
        kept_count = sum(1 for _match, disposition in ordered
                         if disposition == plane_graph.SAME_AXIS_DISPOSITION_KEPT)
        bound = BoundCandidateUniverse(
            frames=frames,
            candidates=tuple(match for match, _disposition in ordered),
            dispositions=tuple(disposition for _match, disposition in ordered),
            kept_count=kept_count,
            dropped_count=len(ordered) - kept_count,
            session_sha256=trusted_session_sha256,
            graph_sha256=trusted_graph_sha256,
            universe_sha256=digest)
        return CandidateUniverseBinding("bound", "", bound)
    except _Refusal as error:
        return CandidateUniverseBinding("refused", str(error))
    except (AttributeError, TypeError, ValueError, OverflowError, RecursionError):
        return CandidateUniverseBinding("refused", "invalid_input")


def search_bound_same_axis_candidates(
    readings: tuple[PlaneReading, ...],
    observation: plane_graph.SameAxisCandidateObservation,
    *,
    connection_graph: ImageDerivedConnectionGraph,
    trusted_session_sha256: str,
    trusted_graph_sha256: str,
    trusted_universe_sha256: str | None = None,
    caps: SearchCaps = HARD_CAPS,
) -> SearchResult:
    """Bind one raw observation, then run the unchanged exhaustive search.

    The only candidate input to the search is the validated bound universe, which
    forwards every kept and dropped candidate. Existing caps, atomic refusals,
    rival retention and the unconditional ``reconstruction_authorized=False`` are
    unchanged; this adapter grants no geometry or production authority.
    """
    binding = bind_same_axis_candidate_universe(
        readings, observation, connection_graph=connection_graph,
        trusted_session_sha256=trusted_session_sha256,
        trusted_graph_sha256=trusted_graph_sha256,
        trusted_universe_sha256=trusted_universe_sha256, caps=caps)
    if binding.status != "bound" or binding.universe is None:
        return SearchResult("refused", binding.reason)
    universe = binding.universe
    result = search_same_axis_candidates(
        readings, universe.candidates, connection_graph=connection_graph,
        trusted_session_sha256=trusted_session_sha256,
        trusted_graph_sha256=trusted_graph_sha256, caps=caps)
    if result.status == "refused":
        return result
    return replace(result, universe_sha256=universe.universe_sha256)


def _constraints(graph, frames, session_pin, graph_pin):
    if type(graph) is not ImageDerivedConnectionGraph:
        raise _Refusal("unbound_constraints")
    if (graph.status != "ok" or graph.session_sha256 != session_pin
            or graph.graph_sha256 != graph_pin
            or tuple(sorted(graph.frame_ids)) != frames):
        raise _Refusal("unbound_constraints")
    # Re-run existing immutable contracts including each nested digest. No
    # session, pixels, matcher, graph builder or inference entry is invoked.
    ConnectionBudgets(**{f.name: getattr(graph.budgets, f.name)
                         for f in fields(ConnectionBudgets)})
    for record in graph.evidence:
        if type(record) is not RelativeCameraEvidence:
            raise _Refusal("unbound_constraints")
        for hypothesis in record.hypotheses:
            YawClassDiagnostic(**{f.name: getattr(hypothesis, f.name)
                                  for f in fields(YawClassDiagnostic)})
        RelativeCameraEvidence(**{f.name: getattr(record, f.name)
                                  for f in fields(RelativeCameraEvidence)})
        if record.session_sha256 != session_pin:
            raise _Refusal("unbound_constraints")
        if record.status == "solved" and (
                type(record.right_yaw_class) is not int
                or record.right_yaw_class not in (0, 1, 2, 3)):
            raise _Refusal("invalid_constraints")
    ImageDerivedConnectionGraph(**{f.name: getattr(graph, f.name)
                                    for f in fields(ImageDerivedConnectionGraph)})
    edges = tuple((r.left_id, r.right_id, r.right_yaw_class)
                  for r in graph.evidence if r.status == "solved")
    if not edges:
        raise _Refusal("empty_constraints")
    adjacency = {frame: [] for frame in frames}
    for left, right, relative in edges:
        adjacency[left].append((right, relative))
        adjacency[right].append((left, (-relative) % 4))
    components = []
    potentials = {}
    for root in frames:
        if root in potentials:
            continue
        potentials[root] = 0
        pending = [root]
        component = []
        while pending:
            current = pending.pop()
            component.append(current)
            for other, relative in adjacency[current]:
                expected = (potentials[current] + relative) % 4
                if other not in potentials:
                    potentials[other] = expected
                    pending.append(other)
                elif potentials[other] != expected:
                    raise _Refusal("constraint_contradiction")
        components.append(tuple(sorted(component)))
    return edges, tuple(components), potentials


def _require_solver_finite(value: object) -> None:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise _Refusal("invalid_solver_output")


def _validate_solver_output(solution, frames, quadrants) -> None:
    """Reject invalid numeric diagnostics before they can become an outcome.

    NumPy error-state guards cover intermediate NumPy/objective arithmetic;
    explicit validation also covers LAPACK results and Python-float arithmetic
    that need not emit a NumPy warning. No numerical solver policy is changed.
    """
    if (type(solution) is not plane_graph.ComponentSolution
            or solution.frames != frames or dict(solution.quadrants) != quadrants
            or set(solution.translations) != set(frames)
            or type(solution.scale_known) is not bool
            or solution.reason not in ("", *plane_graph.COMPONENT_ABSTENTION_REASONS)):
        raise _Refusal("invalid_solver_output")
    for values in solution.translations.values():
        if type(values) is not tuple or len(values) != 3:
            raise _Refusal("invalid_solver_output")
        for value in values:
            _require_solver_finite(value)
    for mapping in (solution.axis_residuals, solution.axis_resultants,
                    solution.blocks_per_period):
        for value in mapping.values():
            _require_solver_finite(value)
    if type(solution.alternatives) is not tuple:
        raise _Refusal("invalid_solver_output")
    for alternative in solution.alternatives:
        if type(alternative) is not plane_graph.ComponentAlternative:
            raise _Refusal("invalid_solver_output")
        _require_solver_finite(alternative.cost)
        for value in alternative.resultants:
            _require_solver_finite(value)
        for value in alternative.axis_observations:
            _require_solver_finite(value)
        if (len(alternative.quadrants) != len(frames)
                or any(type(value) is not int or value not in (0, 1, 2, 3)
                       for value in alternative.quadrants)):
            raise _Refusal("invalid_solver_output")


def search_same_axis_candidates(
    readings: tuple[PlaneReading, ...],
    candidates: tuple[plane_graph.PlaneMatch, ...],
    *,
    connection_graph: ImageDerivedConnectionGraph,
    trusted_session_sha256: str,
    trusted_graph_sha256: str,
    caps: SearchCaps = HARD_CAPS,
) -> SearchResult:
    """Exhaust all feasible nonempty subsets, retaining every full-axis outcome.

    Search nodes include each quadrant assignment and each binary subset-tree
    node. The conservative preflight reserves the *entire* tree and maximum
    possible solver calls/outcomes before enumeration. Caps may only decrease.
    Incompatible quadrants, mixed endpoint b values and missing-axis subsets
    are infeasible leaves. Solver refusals on otherwise feasible leaves remain
    diagnostic rivals. Any cap/fault/contradiction returns no partial outcomes.
    """
    nodes = calls = 0
    try:
        if type(caps) is not SearchCaps:
            raise _Refusal("invalid_caps")
        for field in fields(caps):
            value = getattr(caps, field.name)
            if type(value) is not int or not 0 < value <= getattr(HARD_CAPS, field.name):
                raise _Refusal("invalid_caps")
        if type(readings) is not tuple or type(candidates) is not tuple:
            raise _Refusal("invalid_input")
        if len(candidates) > caps.candidate_identities:
            raise _Refusal("candidate_identities_cap")
        if len(readings) > caps.readings:
            raise _Refusal("readings_cap")
        # Measure before canonical indexing/digest replay (which can allocate).
        representation_bytes((readings, candidates, connection_graph), caps.diagnostic_bytes)
        index, _, frames = plane_graph._canonical_reading_index(readings)
        if len(frames) > caps.frames:
            raise _Refusal("frames_cap")
        if not candidates or len(frames) < 2:
            raise _Refusal("no_candidates")
        identities = {}
        for match in candidates:
            plane_graph._validate_indexed_match(match, index, require_selected_b=True)
            plane_graph.PlaneMatch(**{f.name: getattr(match, f.name)
                                     for f in fields(plane_graph.PlaneMatch)})
            if (match.left >= match.right
                    or type(match.quadrant_of_right) is not int
                    or match.quadrant_of_right not in (0, 1, 2, 3)
                    or plane_graph.axis_transform(match.right_axis, match.quadrant_of_right)[0]
                    != match.left_axis
                    or match.integer_residual is not None
                    or type(match.overlap_cells) is not int or match.overlap_cells < 0):
                raise _Refusal("invalid_candidate")
            for numeric in (match.psr, match.peak, match.second_peak, match.score,
                            match.left_b, match.right_b):
                if type(numeric) not in (float, int) or not math.isfinite(numeric):
                    raise _Refusal("invalid_candidate")
            key = candidate_identity(match)
            if key in identities:
                raise _Refusal("duplicate_identity" if identities[key] == match
                               else "conflicting_identity")
            identities[key] = match
        edges, components, potentials = _constraints(
            connection_graph, frames, trusted_session_sha256, trusted_graph_sha256)
        ordered = tuple(identities[key] for key in sorted(identities))
        quadrant_count = 4 ** (len(components) - 1)
        leaf_count = 2 ** len(ordered) - 1
        reserved_nodes = quadrant_count * (2 ** (len(ordered) + 1))
        reserved_calls = quadrant_count * leaf_count
        if reserved_nodes > caps.search_nodes:
            raise _Refusal("search_nodes_cap")
        if reserved_calls > caps.solver_calls:
            raise _Refusal("solver_calls_cap")
        if reserved_calls > caps.retained_outcomes:
            raise _Refusal("retained_outcomes_cap")
        retained = []
        any_axes = False
        for offsets in product(range(4), repeat=len(components) - 1):
            nodes += 1
            qmap = {frame: (potentials[frame] + offset) % 4
                    for component, offset in zip(components, (0, *offsets))
                    for frame in component}
            compatible = tuple(match for match in ordered
                               if (qmap[match.right] - qmap[match.left]) % 4
                               == match.quadrant_of_right)
            if {plane_graph.axis_transform(m.left_axis, qmap[m.left])[0]
                    for m in compatible} == {0, 1, 2}:
                any_axes = True
            # Traverse the full original tree, even for incompatible records;
            # accounting and returned identity order do not depend on pruning.
            def walk(position, selected):
                nonlocal nodes, calls
                nodes += 1
                if position < len(ordered):
                    walk(position + 1, selected)
                    walk(position + 1, (*selected, ordered[position]))
                    return
                if not selected or any(m not in compatible for m in selected):
                    return
                axes = {plane_graph.axis_transform(m.left_axis, qmap[m.left])[0]
                        for m in selected}
                if axes != {0, 1, 2}:
                    return
                b_values = {}
                for match in selected:
                    for endpoint, b in (((match.left, match.left_plane), match.left_b),
                                        ((match.right, match.right_plane), match.right_b)):
                        if endpoint in b_values and b_values[endpoint] != b:
                            return
                        b_values[endpoint] = b
                fixed = tuple((frames[0], frame, qmap[frame]) for frame in frames[1:])
                # Python float multiply/subtract can overflow silently. Check
                # the exact signed-face and target arithmetic before LAPACK.
                for match in selected:
                    left = index[(match.left, match.left_plane)]
                    right = index[(match.right, match.right_plane)]
                    left_sign = plane_graph.axis_transform(left.axis, qmap[match.left])[1]
                    right_sign = plane_graph.axis_transform(right.axis, qmap[match.right])[1]
                    left_face = left_sign * left.side * left.distance_for(match.left_b)
                    right_face = right_sign * right.side * right.distance_for(match.right_b)
                    for value in (left_face, right_face, left_face - right_face):
                        _require_solver_finite(value)
                calls += 1
                try:
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        solution = plane_graph.solve_component(
                            readings, selected,
                            quadrant_constraints=tuple(sorted(set(edges + fixed))))
                    _validate_solver_output(solution, frames, qmap)
                except (FloatingPointError, OverflowError, np.linalg.LinAlgError):
                    raise _Refusal("invalid_solver_output") from None
                if solution.reason in ("component_too_large", "quadrant_constraint_inconsistent"):
                    raise _Refusal("solver_contradiction")
                outcome = CandidateOutcome(
                    tuple(candidate_identity(m) for m in selected), solution.frames,
                    tuple(sorted(solution.quadrants.items())),
                    tuple(sorted(solution.translations.items())),
                    tuple(sorted(solution.axis_residuals.items())),
                    tuple(sorted(solution.axis_resultants.items())),
                    tuple(sorted(solution.blocks_per_period.items())), solution.reason,
                    solution.scale_known, solution.alternatives)
                retained.append(outcome)
                representation_bytes((ordered, tuple(retained)), caps.diagnostic_bytes)
            walk(0, ())
        if not any_axes:
            raise _Refusal("missing_required_axes")
        if not any(outcome.solver_reason == "" for outcome in retained):
            raise _Refusal("no_consistent_assignment")
        outcomes = tuple(sorted(retained, key=lambda r: (r.quadrants, r.identities)))
        result = SearchResult("provisional_ambiguous" if len(outcomes) > 1
                              else "provisional_unique", "", outcomes, ordered, nodes, calls)
        measured = representation_bytes(result, caps.diagnostic_bytes)
        # Both integer slots have identical allocated size at these hard caps.
        result = replace(result, diagnostic_bytes=measured)
        measured = representation_bytes(result, caps.diagnostic_bytes)
        return replace(result, diagnostic_bytes=measured)
    except _Refusal as error:
        return SearchResult("refused", str(error), search_nodes=nodes, solver_calls=calls)
    except (AttributeError, TypeError, ValueError, OverflowError, RecursionError):
        return SearchResult("refused", "invalid_input", search_nodes=nodes, solver_calls=calls)
