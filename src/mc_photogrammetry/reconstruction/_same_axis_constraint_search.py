"""Versioned synthetic integrity binding and bounded exact diagnostic search.

Synthetic envelope pins test integrity only. A caller who omits rows and seals
a new envelope has authenticated no real producer completeness. No production
caller, geometry selector, or reconstruction authority is supplied here.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
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
from mc_photogrammetry.reconstruction import _same_axis_candidate_search as oracle
from mc_photogrammetry.reconstruction.lattice_planes import (
    DistanceAlternative, PlaneReading, reading_order_key,
)


SCHEMA = "mc-photogrammetry-typed-synthetic-constraint-universe-v1"
CAP_VERSION = "typed-synthetic-high-cap-v1"
SOURCE_KIND = "synthetic_declared_complete"
ENVELOPE_DOMAIN = "mc-photogrammetry-synthetic-producer-envelope-v1"


@dataclass(frozen=True, slots=True)
class ConstraintCaps:
    candidate_identities: int = 4096
    readings: int = 256
    frames: int = 32
    alternatives_per_reading: int = 64
    canonical_input_bytes: int = 16 * 1024 * 1024
    visited_nodes: int = 262144
    candidate_incidence_checks: int = 100_000_000
    numerical_evaluations: int = 8192
    retained_outcomes: int = 512
    memo_bytes: int = 64 * 1024 * 1024
    diagnostic_bytes: int = 16 * 1024 * 1024


HARD_CAPS = ConstraintCaps()


@dataclass(frozen=True, slots=True)
class WorkCounters:
    visited_nodes: int = 0
    candidate_incidence_checks: int = 0
    numerical_evaluations: int = 0
    retained_outcomes: int = 0
    canonical_input_bytes: int = 0
    memo_bytes: int = 0
    diagnostic_bytes: int = 0


class _Refusal(Exception):
    pass


class _Budget:
    """Pre-debited logical work and conservative allocation reservations.

    memo_bytes accounts cumulatively for workspace, trails, digest temporaries
    and bounded validation, not just a cache. No memo cache is needed. Input and
    evidence byte counts are conservative reservation bounds, not RSS/CPU claims.
    A failed debit is not consumed. No partial evidence escapes any refusal.
    """

    def __init__(self, caps):
        if type(caps) is not ConstraintCaps:
            raise _Refusal("invalid_caps")
        for field in fields(caps):
            value = getattr(caps, field.name)
            if type(value) is not int or not 0 < value <= getattr(HARD_CAPS, field.name):
                raise _Refusal("invalid_caps")
        self.caps = caps
        self.values = {field.name: 0 for field in fields(WorkCounters)}

    def debit(self, name, count=1):
        if count < 0 or self.values[name] + count > getattr(self.caps, name):
            raise _Refusal(name + "_cap")
        self.values[name] += count

    def check(self, count=1):
        self.debit("candidate_incidence_checks", count)

    def allocate(self, count):
        self.debit("memo_bytes", count)

    def snapshot(self):
        return WorkCounters(**self.values)


@dataclass(frozen=True, slots=True)
class SyntheticEnvelope:
    sha256: str
    candidate_count: int
    reading_count: int
    graph_frame_count: int
    kept_count: int
    dropped_count: int


@dataclass(frozen=True, slots=True)
class BoundConstraintUniverse:
    schema: str
    cap_version: str
    source_kind: str
    readings: tuple[PlaneReading, ...]
    candidates: tuple[plane_graph.PlaneMatch, ...]
    dispositions: tuple[str, ...]
    connection_graph: ImageDerivedConnectionGraph
    session_sha256: str
    graph_sha256: str
    envelope: SyntheticEnvelope
    frames: tuple[str, ...]
    omitted_frame_ids: tuple[str, ...]
    projected_relations: tuple[tuple[str, str, int], ...]
    universe_sha256: str
    reconstruction_authorized: bool = False


@dataclass(frozen=True, slots=True)
class BindingResult:
    status: str
    reason: str = ""
    universe: BoundConstraintUniverse | None = None
    envelope: SyntheticEnvelope | None = None
    counters: WorkCounters = WorkCounters()
    reconstruction_authorized: bool = False


@dataclass(frozen=True, slots=True)
class ConstraintOutcome:
    oracle_outcome: oracle.CandidateOutcome
    omitted_identities: tuple[tuple, ...]
    axis_ranks: tuple[tuple[int, int], ...]
    equation_residuals: tuple[tuple[int, tuple[float, ...]], ...]
    cycle_sums: tuple[tuple[int, tuple[float, ...]], ...]
    reconstruction_authorized: bool = False


@dataclass(frozen=True, slots=True)
class ConstraintSearchResult:
    status: str
    reason: str
    outcomes: tuple[ConstraintOutcome, ...] = ()
    candidates: tuple[plane_graph.PlaneMatch, ...] = ()
    counters: WorkCounters = WorkCounters()
    universe_sha256: str = ""
    reconstruction_authorized: bool = False


def _finite(value, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (positive and value <= 0)):
        raise _Refusal("invalid_input")


def _solver_finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise _Refusal("invalid_solver_output")


def _identity_reservation(identity):
    # Identity shape is fixed and validated. Account large frame IDs and large
    # shift integers before retained evidence can reference/serialize them.
    total = 128
    for value in identity:
        values = value if type(value) is tuple else (value,)
        for scalar in values:
            total += 6 * len(scalar) + 2 if type(scalar) is str else 6 * sys.getsizeof(scalar)
    return max(512, total)


def _shape_preflight(readings, observation, graph, budget,
                     observation_type=plane_graph.SameAxisCandidateObservation):
    """Check cardinalities before candidate-sized copying, indexing or sorting."""
    budget.check()
    if type(readings) is not tuple:
        raise _Refusal("invalid_input")
    if type(observation) is not observation_type:
        raise _Refusal("invalid_observation")
    if type(observation.candidates) is not tuple or type(observation.dispositions) is not tuple:
        raise _Refusal("invalid_observation")
    if len(observation.candidates) > budget.caps.candidate_identities:
        raise _Refusal("candidate_identities_cap")
    if len(readings) > budget.caps.readings:
        raise _Refusal("readings_cap")
    if len(observation.candidates) != len(observation.dispositions):
        raise _Refusal("unpaired_observation")
    if type(graph) is not ImageDerivedConnectionGraph:
        raise _Refusal("unbound_constraints")
    if type(graph.frame_ids) is not tuple:
        raise _Refusal("unbound_constraints")
    if len(graph.frame_ids) > budget.caps.frames:
        raise _Refusal("frames_cap")
    if type(graph.evidence) is not tuple or len(graph.evidence) > 496:
        raise _Refusal("invalid_constraints")
    for reading in readings:
        budget.check()
        if type(reading) is not PlaneReading or type(reading.alternatives) is not tuple:
            raise _Refusal("invalid_input")
        if len(reading.alternatives) > budget.caps.alternatives_per_reading:
            raise _Refusal("alternatives_per_reading_cap")


def _scan(value, budget):
    """Iterative, bounded typed traversal before any canonical workspace.

    Reserve an upper JSON size (ASCII string escaping <=6 bytes/codepoint) and
    resident representation size; reject cycles/depth and foreign containers.
    This avoids recursive payload traversal and unbounded dataclass asdict.
    """
    budget.allocate(1024)
    stack = [(iter((value,)), 0)]
    size = 0
    while stack:
        iterator, depth = stack[-1]
        try:
            item = next(iterator)
        except StopIteration:
            stack.pop()
            continue
        budget.check()
        if depth > 24:
            raise _Refusal("invalid_input")
        size += max(sys.getsizeof(item), 32)
        if type(item) is str:
            size += 6 * len(item) + 2
        if size > budget.caps.canonical_input_bytes:
            raise _Refusal("canonical_input_bytes_cap")
        if item is None or type(item) in (str, bool):
            continue
        if type(item) in (int, float):
            _finite(item)
        elif type(item) is tuple:
            budget.allocate(128)
            stack.append((iter(item), depth + 1))
        elif is_dataclass(item) and not isinstance(item, type):
            if not item.__dataclass_params__.frozen:
                raise _Refusal("invalid_input")
            budget.allocate(256)
            stack.append((iter(tuple(getattr(item, f.name) for f in fields(item))), depth + 1))
        else:
            raise _Refusal("invalid_input")
    # Charge the raw typed input bound; _prepare reserves canonical expansion.
    budget.debit("canonical_input_bytes", size)
    budget.allocate(size * 4)
    return size


def _primitive(value, budget):
    """Iterative conversion to plain JSON lists/maps; scan reserved its space."""
    budget.allocate(256)
    holder = [None]
    stack = [(value, holder, 0)]
    while stack:
        item, parent, key = stack.pop()
        budget.check()
        if is_dataclass(item) and not isinstance(item, type):
            names = fields(item)
            budget.allocate(128 + 96 * len(names))
            child = {}
            parent[key] = child
            for field in reversed(names):
                stack.append((getattr(item, field.name), child, field.name))
        elif type(item) is tuple:
            budget.allocate(64 + 48 * len(item))
            child = [None] * len(item)
            parent[key] = child
            for i in range(len(item) - 1, -1, -1):
                stack.append((item[i], child, i))
        else:
            parent[key] = item
    return holder[0]


def _digest(domain, payload, reserve, budget):
    # Reserve the largest possible string token before encoding. Traversal is
    # iterative even for JSON; no recursive walker touches caller payloads.
    budget.allocate(reserve * 2 + 4096)
    try:
        digest = hashlib.sha256()
        digest.update(domain.encode("ascii"))
        stack = [(iter((payload,)), "")]
        consumed = 0
        first = [True]
        while stack:
            budget.check()
            iterator, closing = stack[-1]
            try:
                item = next(iterator)
            except StopIteration:
                stack.pop()
                first.pop()
                token = closing
            else:
                prefix = "" if first[-1] else ","
                first[-1] = False
                if type(item) is list:
                    token = prefix + "["
                    stack.append((iter(item), "]"))
                    first.append(True)
                elif type(item) is dict:
                    budget.check(len(item) * max(1, len(item).bit_length()))
                    # Dictionaries are module-created field maps. A key/value
                    # pair iterator emits flat alternating tokens, preserving
                    # sort_keys canonical JSON without recursion.
                    token = prefix + "{"
                    ordered = sorted(item)
                    stack.append((iter(_json_field_tokens(item, ordered)), "}"))
                    first.append(True)
                elif type(item) is _JsonField:
                    token = prefix + json.dumps(item.key, ensure_ascii=True) + ":"
                    stack.append((iter((item.value,)), ""))
                    first.append(True)
                else:
                    token = prefix + json.dumps(item, ensure_ascii=True, allow_nan=False,
                                                separators=(",", ":"))
            consumed += len(token)
            if consumed > reserve:
                raise _Refusal("canonical_input_bytes_cap")
            digest.update(token.encode("ascii"))
        return digest.hexdigest()
    except _Refusal:
        raise
    except Exception:
        raise _Refusal("digest_fault") from None


@dataclass(frozen=True, slots=True)
class _JsonField:
    key: str
    value: object


def _json_field_tokens(mapping, ordered):
    for key in ordered:
        yield _JsonField(key, mapping[key])


def _revalidate(value, cls, budget):
    budget.check(len(fields(cls)))
    budget.allocate(4096 + 256 * len(fields(cls)))
    return cls(**{f.name: getattr(value, f.name) for f in fields(cls)})


def _camera_projection(graph, frames, session_pin, graph_pin, budget):
    if (graph.status != "ok" or graph.session_sha256 != session_pin
            or graph.graph_sha256 != graph_pin or not set(frames) <= set(graph.frame_ids)):
        raise _Refusal("unbound_constraints")
    if type(graph.budgets) is not ConnectionBudgets:
        raise _Refusal("invalid_constraints")
    _revalidate(graph.budgets, ConnectionBudgets, budget)
    for record in graph.evidence:
        budget.check()
        if type(record) is not RelativeCameraEvidence or record.session_sha256 != session_pin:
            raise _Refusal("unbound_constraints")
        for hypothesis in record.hypotheses:
            if type(hypothesis) is not YawClassDiagnostic:
                raise _Refusal("invalid_constraints")
            _revalidate(hypothesis, YawClassDiagnostic, budget)
        _revalidate(record, RelativeCameraEvidence, budget)
        if record.status == "solved" and (type(record.right_yaw_class) is not int
                                          or record.right_yaw_class not in range(4)):
            raise _Refusal("invalid_constraints")
    _revalidate(graph, ImageDerivedConnectionGraph, budget)
    budget.allocate(4096)
    adjacency = {frame: [] for frame in graph.frame_ids}
    for record in graph.evidence:
        budget.check()
        if record.status == "solved":
            adjacency[record.left_id].append((record.right_id, record.right_yaw_class))
            adjacency[record.right_id].append((record.left_id, -record.right_yaw_class % 4))
    potentials = {}
    components = []
    # Full-domain contradictions are replayed even in components without planes.
    for root in graph.frame_ids:
        budget.check()
        if root in potentials:
            continue
        potentials[root] = 0
        pending = [root]
        members = []
        while pending:
            current = pending.pop()
            members.append(current)
            for other, relative in adjacency[current]:
                budget.check()
                expected = (potentials[current] + relative) % 4
                if other not in potentials:
                    potentials[other] = expected
                    pending.append(other)
                elif potentials[other] != expected:
                    raise _Refusal("constraint_contradiction")
        components.append(tuple(frame for frame in frames if frame in members))
    budget.allocate(2048)
    projected = []
    for component in components:
        if component:
            root = component[0]
            for member in component[1:]:
                budget.check()
                projected.append((root, member, (potentials[member] - potentials[root]) % 4))
    order = {frame: i for i, frame in enumerate(graph.frame_ids)}
    budget.check(len(projected) * max(1, len(projected).bit_length()))
    projected.sort(key=lambda row: (order[row[0]], order[row[1]]))
    omitted = tuple(frame for frame in graph.frame_ids if frame not in frames)
    return tuple(projected), omitted


def _prepare(readings, observation, graph, session_pin, graph_pin, budget):
    _shape_preflight(readings, observation, graph, budget)
    if not oracle._is_sha256(session_pin) or not oracle._is_sha256(graph_pin):
        raise _Refusal("invalid_pins")
    scanned = _scan((readings, observation, graph), budget)
    # Reserve added canonical keys/endpoint rows before any canonical workspace.
    reserve = scanned * 3 + 16384
    if reserve > budget.caps.canonical_input_bytes:
        raise _Refusal("canonical_input_bytes_cap")
    budget.debit("canonical_input_bytes", reserve - scanned)
    # Charge ordering comparisons before the frozen canonical helper allocates.
    budget.check(max(1, len(readings)) * max(1, len(readings).bit_length()) * 64)
    budget.allocate(1024 + 1024 * len(readings))
    index, _, frames = plane_graph._canonical_reading_index(readings)
    canonical_readings = tuple(index[key] for key in sorted(index))
    for reading in canonical_readings:
        budget.check()
        _finite(reading.side)
        if reading.side not in (-1.0, 1.0) or not reading.alternatives:
            raise _Refusal("invalid_reading")
        labels = set()
        for alternative in reading.alternatives:
            budget.check()
            if type(alternative) is not DistanceAlternative:
                raise _Refusal("invalid_reading")
            _finite(alternative.blocks_per_period, positive=True)
            _finite(alternative.distance_blocks, positive=True)
            _finite(alternative.support)
            if alternative.blocks_per_period in labels:
                raise _Refusal("invalid_reading")
            labels.add(alternative.blocks_per_period)
    relations, omitted = _camera_projection(graph, frames, session_pin, graph_pin, budget)
    budget.allocate(1024 + len(observation.candidates) * 1024)
    identities = {}
    rows = []
    for candidate, disposition in zip(observation.candidates, observation.dispositions):
        budget.check(2 * budget.caps.alternatives_per_reading + 32)
        if type(candidate) is not plane_graph.PlaneMatch:
            raise _Refusal("invalid_candidate")
        plane_graph._validate_indexed_match(candidate, index, require_selected_b=True)
        _revalidate(candidate, plane_graph.PlaneMatch, budget)
        if (candidate.left >= candidate.right
                or type(candidate.quadrant_of_right) is not int
                or candidate.quadrant_of_right not in range(4)
                or plane_graph.axis_transform(candidate.right_axis, candidate.quadrant_of_right)[0]
                != candidate.left_axis or candidate.integer_residual is not None
                or type(candidate.overlap_cells) is not int or candidate.overlap_cells < 0):
            raise _Refusal("invalid_candidate")
        for value in (candidate.psr, candidate.peak, candidate.second_peak, candidate.score,
                      candidate.left_b, candidate.right_b):
            _finite(value)
        if disposition not in (plane_graph.SAME_AXIS_DISPOSITION_KEPT,
                               plane_graph.SAME_AXIS_DISPOSITION_DROPPED):
            raise _Refusal("invalid_disposition")
        identity = oracle.candidate_identity(candidate)
        if identity in identities:
            raise _Refusal("duplicate_identity" if identities[identity] == candidate
                           else "conflicting_identity")
        identities[identity] = candidate
        rows.append((candidate, disposition))
    budget.check(max(1, len(rows)) * max(1, len(rows).bit_length()))
    rows.sort(key=lambda row: oracle.candidate_identity(row[0]))
    candidates = tuple(row[0] for row in rows)
    dispositions = tuple(row[1] for row in rows)
    kept = dispositions.count(plane_graph.SAME_AXIS_DISPOSITION_KEPT)
    counts = (len(candidates), len(readings), len(graph.frame_ids), kept, len(candidates) - kept)
    # Every unused reading and its full alternative/order table is sealed too.
    reading_rows = tuple((frame, i, reading_order_key(index[(frame, i)]), index[(frame, i)])
                         for frame, i in sorted(index))
    budget.check(len(candidates) * 2 * budget.caps.alternatives_per_reading)
    endpoint_rows = tuple((oracle.candidate_identity(m),
                           index[(m.left, m.left_plane)].distance_for(m.left_b),
                           index[(m.right, m.right_plane)].distance_for(m.right_b))
                          for m in candidates)
    payload = _primitive((SCHEMA, CAP_VERSION, SOURCE_KIND,
                          tuple(getattr(HARD_CAPS, f.name) for f in fields(HARD_CAPS)),
                          reading_rows, candidates, dispositions, endpoint_rows,
                          graph, session_pin, graph_pin, frames, omitted, relations, counts), budget)
    envelope_digest = _digest(ENVELOPE_DOMAIN, payload, reserve, budget)
    return (canonical_readings, candidates, dispositions, frames, omitted,
            relations, SyntheticEnvelope(envelope_digest, *counts), payload, reserve)


def declare_synthetic_envelope(readings, observation, *, connection_graph,
                               trusted_session_sha256, trusted_graph_sha256,
                               caps=HARD_CAPS):
    """Fixture-only pre-mutation seal; cannot authenticate a real producer."""
    budget = None
    try:
        budget = _Budget(caps)
        prepared = _prepare(readings, observation, connection_graph,
                            trusted_session_sha256, trusted_graph_sha256, budget)
        return BindingResult("sealed", envelope=prepared[6], counters=budget.snapshot())
    except _Refusal as error:
        return BindingResult("refused", str(error), counters=budget.snapshot() if budget else WorkCounters())
    except Exception:
        return BindingResult("refused", "invalid_input", counters=budget.snapshot() if budget else WorkCounters())


def _bind(readings, observation, graph, session_pin, graph_pin, envelope, source_kind,
          expected_digest, budget):
    if source_kind != SOURCE_KIND:
        raise _Refusal("unsupported_provenance")
    if type(envelope) is not SyntheticEnvelope or not oracle._is_sha256(envelope.sha256):
        raise _Refusal("invalid_envelope")
    for name in ("candidate_count", "reading_count", "graph_frame_count", "kept_count", "dropped_count"):
        if type(getattr(envelope, name)) is not int or getattr(envelope, name) < 0:
            raise _Refusal("invalid_envelope")
    prepared = _prepare(readings, observation, graph, session_pin, graph_pin, budget)
    canonical, candidates, dispositions, frames, omitted, relations, actual, payload, reserve = prepared
    if actual != envelope:
        raise _Refusal("producer_envelope_mismatch")
    digest = _digest(SCHEMA, [payload, _primitive(envelope, budget)], reserve + 2048, budget)
    if expected_digest is not None:
        if not oracle._is_sha256(expected_digest):
            raise _Refusal("invalid_pins")
        if expected_digest != digest:
            raise _Refusal("binding_mismatch")
    budget.allocate(1024)
    return BoundConstraintUniverse(SCHEMA, CAP_VERSION, SOURCE_KIND, canonical, candidates,
                                   dispositions, graph, session_pin, graph_pin, envelope, frames,
                                   omitted, relations, digest)


def bind_constraint_universe(readings, observation, *, connection_graph,
                             trusted_session_sha256, trusted_graph_sha256,
                             producer_envelope, source_kind,
                             trusted_universe_sha256=None, caps=HARD_CAPS):
    budget = None
    try:
        budget = _Budget(caps)
        universe = _bind(readings, observation, connection_graph, trusted_session_sha256,
                         trusted_graph_sha256, producer_envelope, source_kind,
                         trusted_universe_sha256, budget)
        return BindingResult("bound", universe=universe, counters=budget.snapshot())
    except _Refusal as error:
        return BindingResult("refused", str(error), counters=budget.snapshot() if budget else WorkCounters())
    except Exception:
        return BindingResult("refused", "invalid_input", counters=budget.snapshot() if budget else WorkCounters())


def _yaw_components(universe, budget):
    budget.allocate(4096)
    adjacency = {frame: [] for frame in universe.frames}
    for left, right, relative in universe.projected_relations:
        budget.check()
        adjacency[left].append((right, relative))
        adjacency[right].append((left, -relative % 4))
    potentials = {}
    components = []
    for root in universe.frames:
        if root in potentials:
            continue
        potentials[root] = 0
        pending = [root]
        members = []
        while pending:
            current = pending.pop()
            members.append(current)
            for other, relative in adjacency[current]:
                budget.check()
                if other not in potentials:
                    potentials[other] = (potentials[current] + relative) % 4
                    pending.append(other)
        components.append(tuple(sorted(members)))
    return tuple(components), potentials


def _diagnostics(selected, qmap, index, solution, frames, budget):
    """Incidence rank and equation/cycle evidence, with no gate or pruning."""
    budget.allocate(8192 + len(selected) * 2048)
    rows = {axis: [] for axis in range(3)}
    for match in selected:
        budget.check(128)
        left, right = index[(match.left, match.left_plane)], index[(match.right, match.right_plane)]
        axis, sign_l = plane_graph.axis_transform(left.axis, qmap[match.left])
        _, sign_r = plane_graph.axis_transform(right.axis, qmap[match.right])
        face_l = sign_l * left.side * left.distance_for(match.left_b)
        face_r = sign_r * right.side * right.distance_for(match.right_b)
        for value in (face_l, face_r, face_l - face_r):
            _solver_finite(value)
        rows[axis].append((match.left, match.right, face_l - face_r))
    ranks, residuals, cycles = [], [], []
    for axis in range(3):
        # Rank of the anchored incidence matrix equals F minus the number of
        # connected components, including isolated frames (no numerical gate).
        adjacency = {frame: [] for frame in frames}
        values = {}
        forest = {frame: [] for frame in frames}
        cycle = []
        for left, right, delta in rows[axis]:
            budget.check(2 * len(frames))
            # Find a prior forest path. Non-tree edges yield signed cycle sums.
            pending = [(left, 0.0)]
            seen = {left}
            implied = None
            while pending:
                current, total = pending.pop()
                if current == right:
                    implied = total
                    break
                for other, step in forest[current]:
                    if other not in seen:
                        seen.add(other)
                        pending.append((other, total + step))
            if implied is None:
                forest[left].append((right, delta))
                forest[right].append((left, -delta))
            else:
                _solver_finite(delta - implied)
                cycle.append(float(delta - implied))
            adjacency[left].append(right)
            adjacency[right].append(left)
        groups = 0
        for root in frames:
            if root in values:
                continue
            groups += 1
            pending = [root]
            values[root] = True
            while pending:
                for other in adjacency[pending.pop()]:
                    budget.check()
                    if other not in values:
                        values[other] = True
                        pending.append(other)
        if rows[axis]:
            ranks.append((axis, len(frames) - groups))
            equation = tuple(float(solution.translations[right][axis]
                                   - solution.translations[left][axis] - delta)
                             for left, right, delta in rows[axis])
            for value in equation:
                _solver_finite(value)
            residuals.append((axis, equation))
            cycles.append((axis, tuple(cycle)))
    return tuple(ranks), tuple(residuals), tuple(cycles)


def _leaf(universe, indices, qmap, index, identities, identity_evidence, retained, budget):
    budget.debit("numerical_evaluations")
    budget.allocate(16384 + len(indices) * 4096 + len(universe.frames) ** 3 * 256)
    selected = tuple(universe.candidates[i] for i in sorted(indices))
    # Validate signed arithmetic before entering unchanged LAPACK/phase gates.
    for match in selected:
        budget.check(128)
        left, right = index[(match.left, match.left_plane)], index[(match.right, match.right_plane)]
        left_face = plane_graph.axis_transform(left.axis, qmap[match.left])[1] * left.side * left.distance_for(match.left_b)
        right_face = plane_graph.axis_transform(right.axis, qmap[match.right])[1] * right.side * right.distance_for(match.right_b)
        for value in (left_face, right_face, left_face - right_face):
            _solver_finite(value)
    fixed = tuple((universe.frames[0], frame, qmap[frame]) for frame in universe.frames[1:])
    # The frozen solver checks its full finite yaw domain even under fixed yaw.
    # Charge those comparisons and its endpoint/matrix work before calling it.
    budget.check(4 ** (len(universe.frames) - 1)
                 * max(1, len(universe.projected_relations) + len(fixed)) * 3
                 + len(selected) * (128 + 3 * len(universe.frames) ** 2))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            solution = plane_graph.solve_component(
                universe.readings, selected,
                quadrant_constraints=tuple(sorted(set(universe.projected_relations + fixed))))
        oracle._validate_solver_output(solution, universe.frames, qmap)
    except Exception:
        raise _Refusal("invalid_solver_output") from None
    if solution.reason in ("component_too_large", "quadrant_constraint_inconsistent"):
        raise _Refusal("solver_contradiction")
    ranks, residuals, cycles = _diagnostics(selected, qmap, index, solution, universe.frames, budget)
    budget.debit("retained_outcomes")
    # Conservative full outcome reservation includes every selected/omitted
    # identity and all diagnostics. Debit before constructing retained evidence.
    evidence = (8192 + identity_evidence + len(selected) * 1024
                + 24 * sum(len(frame) for frame in universe.frames)
                + 2 * sum(_identity_reservation(identities[i]) for i in indices))
    budget.debit("diagnostic_bytes", evidence)
    budget.allocate(evidence)
    old = oracle.CandidateOutcome(
        tuple(oracle.candidate_identity(m) for m in selected), solution.frames,
        tuple(sorted(solution.quadrants.items())), tuple(sorted(solution.translations.items())),
        tuple(sorted(solution.axis_residuals.items())), tuple(sorted(solution.axis_resultants.items())),
        tuple(sorted(solution.blocks_per_period.items())), solution.reason,
        solution.scale_known, solution.alternatives)
    chosen = set(indices)
    omitted = tuple(identity for i, identity in enumerate(identities) if i not in chosen)
    retained.append(ConstraintOutcome(old, omitted, ranks, residuals, cycles))


def search_constraint_universe(universe, *, caps=HARD_CAPS):
    """Exhaust complete yaw/subset rivals or return atomic typed abstention.

    Domains are sparse Boolean candidate domains: omission is always explicit;
    yaw and exact endpoint-b conflicts remove only selection. Branching uses
    smallest domain, then canonical identity. Propagation mutations use a
    reversible trail; stack traversal avoids recursion at 4096 identities.
    No score, rank, support or equation-cycle diagnostic can prune a rival.
    """
    budget = None
    try:
        budget = _Budget(caps)
        if type(universe) is not BoundConstraintUniverse:
            raise _Refusal("invalid_binding")
        if (universe.schema != SCHEMA or universe.cap_version != CAP_VERSION
                or universe.reconstruction_authorized is not False):
            raise _Refusal("invalid_binding")
        _shape_preflight(universe.readings, universe, universe.connection_graph, budget,
                         observation_type=BoundConstraintUniverse)
        budget.check(len(universe.candidates))
        budget.allocate(256)
        observation = plane_graph.SameAxisCandidateObservation(universe.candidates, universe.dispositions)
        replay = _bind(universe.readings, observation, universe.connection_graph,
                       universe.session_sha256, universe.graph_sha256, universe.envelope,
                       universe.source_kind, universe.universe_sha256, budget)
        if replay != universe:
            raise _Refusal("binding_mismatch")
        if not universe.candidates or len(universe.frames) < 2:
            raise _Refusal("no_candidates")
        components, potentials = _yaw_components(universe, budget)
        budget.allocate(1024 + len(universe.readings) * 1024 + len(universe.candidates) * 2048)
        index, _, _ = plane_graph._canonical_reading_index(universe.readings)
        identities = tuple(oracle.candidate_identity(m) for m in universe.candidates)
        budget.check(12 * len(identities))
        identity_evidence = sum(_identity_reservation(identity) for identity in identities)
        # Candidate evidence is part of the final diagnostic too. Reserve it
        # before any leaf or output allocation, including large identity fields.
        budget.debit("diagnostic_bytes", 1024 + identity_evidence + len(identities) * 512)
        endpoints = []
        incidence = {}
        for i, match in enumerate(universe.candidates):
            budget.check(2)
            ends = (((match.left, match.left_plane), match.left_b),
                    ((match.right, match.right_plane), match.right_b))
            endpoints.append(ends)
            for endpoint, b in ends:
                incidence.setdefault(endpoint, []).append((i, b))
        retained = []
        any_axes = False
        for offsets in product(range(4), repeat=len(components) - 1):
            budget.debit("visited_nodes")
            budget.allocate(4096 + len(universe.candidates) * 64)
            qmap = {frame: (potentials[frame] + offset) % 4
                    for members, offset in zip(components, (0, *offsets)) for frame in members}
            domains = []
            axes = []
            for match in universe.candidates:
                budget.check()
                compatible = (qmap[match.right] - qmap[match.left]) % 4 == match.quadrant_of_right
                domains.append(3 if compatible else 1)  # 1=omit, 2=select
                axes.append(plane_graph.axis_transform(match.left_axis, qmap[match.left])[0])
            if {axes[i] for i, domain in enumerate(domains) if domain == 3} != {0, 1, 2}:
                continue
            any_axes = True
            assigned = set()
            selected = set()
            trail = []
            # Each frame is (phase, candidate index, trail checkpoint). A pop
            # restores the parent's domains/assignments before its next branch.
            budget.allocate(256)
            stack = [(0, -1, 0)]
            while stack:
                phase, variable, checkpoint = stack.pop()
                if phase == 2:
                    while len(trail) > checkpoint:
                        pos, old = trail.pop()
                        domains[pos] = old
                    assigned.remove(variable)
                    selected.discard(variable)
                    continue
                if phase == 1:
                    selected.add(variable)
                    for endpoint, b in endpoints[variable]:
                        for other, other_b in incidence[endpoint]:
                            budget.check()
                            if other_b != b and domains[other] & 2:
                                budget.allocate(128)
                                trail.append((other, domains[other]))
                                domains[other] = 1
                    budget.allocate(256)
                    stack.append((2, variable, checkpoint))
                    stack.append((0, -1, 0))
                    continue
                budget.debit("visited_nodes")
                budget.check(len(domains))
                budget.allocate(128 + 32 * len(domains))
                remaining = [i for i, domain in enumerate(domains) if domain == 3 and i not in assigned]
                # Logically impossible three-axis leaves only; preserve the
                # frozen missing_required_axes/no_consistent_assignment order.
                if {axes[i] for i in selected} | {axes[i] for i in remaining} != {0, 1, 2}:
                    continue
                if not remaining:
                    _leaf(universe, selected, qmap, index, identities, identity_evidence,
                          retained, budget)
                    continue
                # Every non-forced Boolean domain has size two. Canonical
                # identity ties therefore choose the smallest canonical index.
                variable = remaining[0]
                assigned.add(variable)
                checkpoint = len(trail)
                budget.allocate(256)
                stack.append((1, variable, checkpoint))
                stack.append((0, -1, 0))
        if not any_axes:
            raise _Refusal("missing_required_axes")
        if not any(outcome.oracle_outcome.solver_reason == "" for outcome in retained):
            raise _Refusal("no_consistent_assignment")
        budget.check(len(retained) * max(1, len(retained).bit_length()))
        budget.allocate(1024 + len(retained) * 64)
        outcomes = tuple(sorted(retained, key=lambda o: (o.oracle_outcome.quadrants,
                                                       o.oracle_outcome.identities)))
        return ConstraintSearchResult("provisional_ambiguous" if len(outcomes) > 1
                                      else "provisional_unique", "", outcomes,
                                      universe.candidates, budget.snapshot(), universe.universe_sha256)
    except _Refusal as error:
        return ConstraintSearchResult("refused", str(error), counters=budget.snapshot() if budget else WorkCounters())
    except Exception:
        return ConstraintSearchResult("refused", "invalid_input", counters=budget.snapshot() if budget else WorkCounters())
