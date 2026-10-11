"""Bounded endpoint CSP. Consistency alone never authorizes geometry.

Each active unordered plane-endpoint pair is mandatory; its existing match
rows contribute an OR relation. No score, PSR, distance or residual ranks a
solution. Complete alternatives must satisfy signed displacement and frame cycles.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from heapq import merge, heappush, heappop
from collections.abc import Sequence
from types import MappingProxyType

from . import plane_graph
from .lattice_planes import PlaneReading, reading_order_key


TRANSCRIPT_POLICY = "source-bound-physical-transcript-v3"
BYTE_CATEGORIES = ("source", "relation", "disposition", "exclusion", "support")
PARTIAL_ARC_COUNTERS = ("arc_relation_visits", "arc_queue_visits", "arc_incidence_visits", "arc_factor_scans")
PHYSICAL_REASON_CODES = {None: 0, "physical_displacement_diameter": 1,
                         "physical_frame_cycle": 2, "unverified_physical_displacement": 3}


@dataclass(frozen=True)
class EndpointBudget:
    endpoint_variables: int = 128
    frames: int = 32
    alternatives_per_endpoint: int = 16
    input_match_rows: int = 16_384
    registry_factors: int = 4_096
    quadrant_assignments: int = 16_384
    branch_nodes: int = 16_384
    relation_visits: int = 262_144
    solution_rows: int = 2
    scalar_payload_bytes: int = 4 * 1024 * 1024
    scalar_row_bytes: int = 65_536

    def __post_init__(self):
        if type(self) not in (EndpointBudget, PartialEndpointBudget):
            raise ValueError("budget requires an exact supported profile")
        for field in fields(EndpointBudget):
            value = getattr(self, field.name)
            ceiling = (524_288 if type(self) is PartialEndpointBudget
                       and field.name == "relation_visits" else field.default)
            if type(value) is not int or not 0 <= value <= ceiling:
                raise ValueError("budgets must be nonnegative integers within hard ceilings")


@dataclass(frozen=True)
class PartialEndpointBudget(EndpointBudget):
    """Explicit partial search only; all other legacy hard ceilings unchanged."""
    relation_visits: int = 524_288


def validate_budget(budget, *, partial=False):
    if type(partial) is not bool:
        raise ValueError("budget mode must be a boolean")
    allowed = (EndpointBudget, PartialEndpointBudget) if partial else (EndpointBudget,)
    if type(budget) not in allowed:
        raise ValueError("budget profile is not allowed for this observability mode")
    type(budget)(**{field.name: getattr(budget, field.name) for field in fields(EndpointBudget)})
    return budget


@dataclass(frozen=True)
class FactorSupport:
    endpoints: tuple[tuple[str, int], tuple[str, int]]
    alternatives: tuple[int, int]
    source_rows: tuple[int, ...]
    distinguishable_support_groups: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class EndpointChoice:
    assignment_index: int
    quadrants: tuple[tuple[str, int], ...]
    alternatives: tuple[tuple[str, int, int], ...]
    factors: tuple[FactorSupport, ...]
    physical_groups: tuple
    physical_relations_digest: str


@dataclass(frozen=True)
class EndpointResult:
    status: str  # no / unique / multiple / unresolved_evidence / budget_refused
    source_digest: str | None
    choices: tuple[EndpointChoice, ...]
    counters: object
    pruning: object
    denial: tuple[str, int, int, int] | None
    exhaustive: bool
    evidence_ambiguous: bool
    reason: str | None
    session_frames: tuple[str, ...]
    eligibility_digest: str
    eligibility_reasons: object
    excluded_assignments: tuple[tuple[int, tuple[tuple[str, int], ...], str], ...]
    physical_policy: str
    physical_tolerance: float
    physical_digest: str
    physical_exclusions: tuple
    partial_exclusions: tuple
    physical_reasons: object
    transcript_policy: str


class _Exhausted(Exception):
    pass


class _Work:
    def __init__(self, budget):
        self.budget = validate_budget(budget, partial=True)
        self.transcript_policy = TRANSCRIPT_POLICY
        self.counts = {field.name: 0 for field in fields(budget)}
        self.counts.update(planned_assignments=None, visited_assignments=0,
                           solutions_found=0, active_factor_visits=0,
                           eligible_assignments=0, excluded_assignments=0,
                           eligibility_hashed_assignments=0,
                           retained_exclusions=0, omitted_exclusions=0)
        self.counts.update(physical_assignments=0, physical_feasible=0,
                           physical_rejected=0, physical_relation_visits=0,
                           physical_cycle_relaxations=0, physical_hashed_assignments=0,
                           retained_physical_exclusions=0, omitted_physical_exclusions=0)
        self.counts.update(partial_checks=0, partial_prunes=0,
                           retained_partial_exclusions=0, omitted_partial_exclusions=0,
                           search_events_attempted=0, search_events_committed=0,
                           search_events_uncommitted=0)
        self.counts.update({name + "_visits": 0 for name in (
            "preflight", "incidence", "partial_support", "partial_update",
            "partial_queue", "partial_journal", "partial_undo", "partial_event")})
        self.counts.update({category + "_bytes": 0 for category in BYTE_CATEGORIES})
        self.counts.update(scalar_records_attempted=0, scalar_records_committed=0,
                           scalar_records_uncommitted=0)
        self.source_digest = None
        self.endpoint_keys = ()
        self.pruning = dict(values_removed=0, inconsistent_nodes=0)
        self.denial = None
        self.session_frames = ()
        self.eligibility_digest = hashlib.sha256()
        self.eligibility_reasons = {}
        self.exclusions = []
        self.physical_digest = hashlib.sha256()
        self.physical_exclusions = []
        self.partial_exclusions = []
        self.physical_reasons = {}

    def bind_source(self, digest, keys):
        self.source_digest, self.endpoint_keys = digest, keys
        self.physical_digest.update(self.encode(
            ["physical-stream", self.transcript_policy, digest], category="disposition"))

    def physical(self, index, quadrants, selected, reason, digest, witness):
        # Q indices address the fixed planner; alternatives address canonical keys.
        row = (index, tuple(selected[key] for key in self.endpoint_keys),
               PHYSICAL_REASON_CODES[reason], digest, witness)
        retained = reason is not None and self.explanation_slots() < 128
        encoded = self.search_event("leaf", row, retained)
        self.physical_digest.update(encoded)
        self.counts["physical_hashed_assignments"] += 1
        if reason is None:
            self.counts["physical_feasible"] += 1
        else:
            self.counts["physical_rejected"] += 1
            self.physical_reasons[reason] = self.physical_reasons.get(reason, 0) + 1
            if retained:
                self.physical_exclusions.append(row)
                self.counts["retained_physical_exclusions"] += 1
            else:
                self.counts["omitted_physical_exclusions"] += 1

    def explanation_slots(self):
        return len(self.physical_exclusions) + len(self.partial_exclusions)

    def search_event(self, kind, row, retained):
        self.counts["search_events_attempted"] += 1
        try:
            encoded = self.encode([kind, row], category="disposition")
            if retained:
                self.encode([kind + "-exclusion", row], category="exclusion")
        except Exception:
            self.counts["search_events_uncommitted"] += 1
            raise
        self.counts["search_events_committed"] += 1
        return encoded

    def partial(self, row):
        retained = self.explanation_slots() < 128
        encoded = self.search_event("partial", row, retained)
        self.physical_digest.update(encoded)
        self.counts["partial_prunes"] += 1
        if retained:
            self.partial_exclusions.append(("partial", *row))
            self.counts["retained_partial_exclusions"] += 1
        else:
            self.counts["omitted_partial_exclusions"] += 1

    def relation_charge(self, attribution, amount=1):
        self.charge("relation_visits", amount)
        self.counts[attribution + "_visits"] += amount

    def enable_partial_arcs(self):
        self.counts.update({name: 0 for name in PARTIAL_ARC_COUNTERS})

    def eligibility(self, index, quadrants, reason):
        row = (index, tuple(quadrants.items()), reason)
        encoded = self.encode(row, category="disposition")
        retained = reason is not None and len(self.exclusions) < 128
        if retained:
            self.encode(["eligibility-exclusion", row], category="exclusion")
        self.eligibility_digest.update(encoded)
        self.counts["eligibility_hashed_assignments"] += 1
        if reason is None:
            self.counts["eligible_assignments"] += 1
        else:
            self.counts["excluded_assignments"] += 1
            self.eligibility_reasons[reason] = self.eligibility_reasons.get(reason, 0) + 1
            if len(self.exclusions) < 128:
                self.exclusions.append(row)
                self.counts["retained_exclusions"] += 1
            else:
                self.counts["omitted_exclusions"] += 1

    def charge(self, name, amount):
        used, cap = self.counts[name], getattr(self.budget, name)
        if used + amount > cap:
            self.denial = (name, used, amount, cap)
            raise _Exhausted
        self.counts[name] += amount

    def bound(self, name, amount):
        cap = getattr(self.budget, name)
        if amount > cap:
            self.denial = (name, 0, amount, cap)
            raise _Exhausted
        self.counts[name] = max(self.counts[name], amount)

    def encode(self, value, *, category="source"):
        if category not in BYTE_CATEGORIES:
            raise ValueError("unknown scalar accounting category")
        self.counts["scalar_records_attempted"] += 1
        try:
            result = self._encode(value, category)
        except Exception:
            self.counts["scalar_records_uncommitted"] += 1
            raise
        self.counts["scalar_records_committed"] += 1
        return result

    def _encode(self, value, category):
        # Bound traversal and strings before allocating JSON chunks or a stack.
        stack, items, lower = [(value, 0)], 0, 0
        cap = self.budget.scalar_row_bytes
        while stack:
            item, depth = stack.pop(); items += 1
            if depth > 16 or items + lower > cap:
                self.bound("scalar_row_bytes", cap + 1)
            if isinstance(item, (tuple, list, dict)):
                self.bound("scalar_row_bytes", len(item) + len(stack) + items + lower)
                children = item.values() if isinstance(item, dict) else item
                if isinstance(item, dict):
                    lower += sum(len(key) for key in item)
                stack.extend((child, depth + 1) for child in children)
            elif isinstance(item, str):
                lower += len(item)
                self.bound("scalar_row_bytes", items + lower)
        data = bytearray()
        encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"),
                                   allow_nan=False, default=lambda value: value.item())
        for chunk in encoder.iterencode(value):
            raw = chunk.encode("utf-8")
            self.bound("scalar_row_bytes", len(data) + len(raw))
            self.charge("scalar_payload_bytes", len(raw))
            self.counts[category + "_bytes"] += len(raw)
            data.extend(raw)
        # Versioned category tag and unsigned length delimit every canonical record.
        # Charge the header too; a partial payload never commits a digest event.
        self.bound("scalar_row_bytes", len(data) + 6)
        self.charge("scalar_payload_bytes", 6)
        self.counts[category + "_bytes"] += 6
        return (b"R" + bytes((BYTE_CATEGORIES.index(category),))
                + len(data).to_bytes(4, "big") + bytes(data))


def _row_payload(row):
    # Retain the exact input evidence, including score, without using it to rank.
    result = {name: getattr(row, name) for name in row.__dataclass_fields__}
    for name, value in result.items():
        if isinstance(value, float) and value != value:
            result[name] = "nan"
        elif isinstance(value, float) and abs(value) == float("inf"):
            result[name] = str(value)
    return result


def _preflight_relations(work, indexed, matches, frames):
    """Validate every original row/pair/orientation before search can close it."""
    for i, row in enumerate(matches):
        work.relation_charge("preflight")
        u, v = (row.left, row.left_plane), (row.right, row.right_plane)
        left, right = indexed[u], indexed[v]
        work.relation_charge("preflight", len(left.alternatives) + len(right.alternatives))
        il = left.matching_alternatives(row.left_b)
        ir = right.matching_alternatives(row.right_b)
        for a, alternative in enumerate(left.alternatives):
            if not any(alternative is item for item in il): continue
            for b, other in enumerate(right.alternatives):
                if not any(other is item for item in ir): continue
                work.relation_charge("preflight")
                for absolute in range(4):
                    work.relation_charge("preflight")
                    q = dict.fromkeys(frames, 0)
                    q[row.left], q[row.right] = absolute, (absolute + row.quadrant_of_right) % 4
                    try:
                        plane_graph.signed_physical_relation(indexed, row, q, {u: a, v: b},
                                                             i, compact=True)
                    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
                        raise ValueError("unverified physical displacement fails closed globally") from error


def _factor_activation_index(work, registry):
    """Bounded invocation-local buckets sharing the exact source relations.

    Canonical ranks retain sorted-registry factor/observer order across buckets.
    Each registry entry and frame-pair index entry is charged before retention.
    """
    work.bound("registry_factors", len(registry))
    grouped = {}
    for rank, ((u, v, relative), relation) in enumerate(sorted(registry.items())):
        work.charge("relation_visits", 1)
        pair = (u[0], v[0])
        if pair not in grouped:
            grouped[pair] = [[], [], [], []]
        grouped[pair][relative].append((rank, (u, v, relation)))
    indexed = []
    for pair in sorted(grouped):
        work.charge("relation_visits", 1)
        indexed.append((pair, grouped[pair]))
    return tuple(indexed)


def _charged_factor_bucket(work, bucket):
    for entry in bucket:
        work.charge("relation_visits", 1)
        yield entry


def _active_factors(work, index, quadrants):
    """Visit only the actual relative-Q bucket, preserving canonical order.

    The merge heap has at most one entry per bounded frame pair; total retained
    bucket entries never exceed registry_factors. No relation is copied/pruned.
    """
    streams = []
    for (left, right), buckets in index:
        work.charge("relation_visits", 1)
        relative = (quadrants[right] - quadrants[left]) % 4
        streams.append(_charged_factor_bucket(work, buckets[relative]))
    return [factor for _, factor in merge(*streams, key=lambda entry: entry[0])]


class _ForcedDiameter:
    """One Q's singleton incidence queue and branch-local reversible extrema."""
    def __init__(self, work, factors, indexed, matches, quadrants):
        self.work, self.factors, self.indexed = work, factors, indexed
        self.matches, self.quadrants = matches, quadrants
        self.incidence = {key: [] for key in work.endpoint_keys}
        self.positions = {key: i for i, key in enumerate(work.endpoint_keys)}
        self.active, self.extrema, self.journal = set(), {}, []
        for fid, (u, v, _) in enumerate(factors):
            for key in (u, v):
                work.relation_charge("incidence")
                self.incidence[key].append(fid)

    def singleton(self, key, pending):
        for fid in self.incidence[key]:
            self.work.relation_charge("incidence")
            if fid not in self.active and fid not in pending:
                self.work.relation_charge("partial_queue")
                pending.add(fid)

    @staticmethod
    def merge(previous, delta, reference):
        if previous is None:
            return (delta, delta, reference, reference)
        low, high, lo, hi = previous
        if delta < low or (delta == low and reference[0] < lo[0]): low, lo = delta, reference
        if delta > high or (delta == high and reference[0] < hi[0]): high, hi = delta, reference
        return (low, high, lo, hi)

    def check(self, pending, domains, index, path):
        while pending:
            fid = min(pending)
            self.work.relation_charge("partial_queue")
            pending.remove(fid)
            u, v, relation = self.factors[fid]
            if fid in self.active or len(domains[u]) != 1 or len(domains[v]) != 1:
                continue
            self.work.counts["partial_checks"] += 1
            a, b = next(iter(domains[u])), next(iter(domains[v]))
            self.work.relation_charge("partial_support")
            rows = relation.get((a, b))
            if not rows:
                raise ValueError("forced factor lacks genuine source support")
            local = {}
            for i in rows:
                self.work.relation_charge("partial_support")
                key, delta, row_id, left, right = plane_graph.signed_physical_relation(
                    self.indexed, self.matches[i], self.quadrants, {u: a, v: b}, i, compact=True)
                reference = (row_id, left, right, key[2], delta)
                self.work.relation_charge("partial_update")
                local[key] = self.merge(local.get(key), delta, reference)
            # Each factor has one canonical frame pair / mapped world axis.
            if len(local) != 1:
                raise ValueError("forced factor physical group differs")
            key, group = next(iter(local.items()))
            previous = self.extrema.get(key)
            self.work.relation_charge("partial_journal")
            self.journal.append((fid, key, previous))
            self.work.relation_charge("partial_update")
            merged = self.merge(previous, group[0], group[2])
            self.extrema[key] = self.merge(merged, group[1], group[3])
            self.work.relation_charge("partial_update")
            self.active.add(fid)
            low, high, lo, hi = self.extrema[key]
            self.work.relation_charge("partial_update")
            if high - low > plane_graph.SAME_AXIS_TOLERANCE:
                forced = []
                for activated in sorted(self.active):
                    self.work.relation_charge("partial_event")
                    fu, fv, support = self.factors[activated]
                    fa, fb = next(iter(domains[fu])), next(iter(domains[fv]))
                    forced.append((activated, self.positions[fu], fa, self.positions[fv], fb,
                                   tuple(support[(fa, fb)])))
                masks = []
                for endpoint in self.work.endpoint_keys:
                    self.work.relation_charge("partial_event")
                    masks.append(sum(1 << value for value in domains[endpoint]))
                self.work.partial((index, path, tuple(masks), tuple(forced), (key, low, high, lo, hi)))
                self.work.pruning["inconsistent_nodes"] += 1
                return False
        return True

    def undo(self, marker):
        while len(self.journal) > marker:
            self.work.relation_charge("partial_undo")
            fid, key, previous = self.journal.pop()
            self.work.relation_charge("partial_undo")
            self.active.remove(fid)
            self.work.relation_charge("partial_undo")
            if previous is None: del self.extrema[key]
            else: self.extrema[key] = previous


class _EndpointIncidenceHooks:
    """Transient, bounded scalar projection; never participates in solver work."""
    MAX_ASSIGNMENTS = 64
    MAX_BYTES = 4 * 1024 * 1024

    def __init__(self, sink):
        self.sink = sink
        self.relations = {}
        self.complete = True
        self.registry_complete = False
        self.bytes = 0
        self.observed = dict(visited=0, entered=0, returned=0, classified=0)

    def capture(self, event, *args):
        if not self.complete:
            return
        try:
            getattr(self, "_" + event)(*args)
        except BaseException:
            # Even a hostile sink cannot change the result/exception of the solve.
            self.complete = False
            self.relations.clear()
            self._safe_emit("incomplete", {"reason": "observer_failure_or_cap"})

    def _safe_emit(self, event, payload):
        try:
            self._emit(event, payload)
        except BaseException:
            self.complete = False

    def _emit(self, event, payload):
        encoder = json.JSONEncoder(ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        size = 0
        for chunk in encoder.iterencode((event, payload)):
            size += len(chunk.encode("ascii"))
            if size + self.bytes > self.MAX_BYTES:
                raise ValueError("observer byte ceiling")
        self.bytes += size
        self.sink(event, MappingProxyType(payload))

    @staticmethod
    def _frame(value):
        if (type(value) is not str or not 1 <= len(value) <= 128
                or not value.isascii()
                or any(not (c.isalnum() or c in "_-.") for c in value)
                or value in (".", "..")):
            raise ValueError("observer frame label")
        return value

    def _registry(self, registry, indexed, rows, source_digest):
        if len(indexed) > 128 or len(registry) > 4096 or rows > 16384:
            raise ValueError("observer registry ceiling")
        endpoints = []
        for (frame, index), reading in sorted(indexed.items()):
            self._frame(frame)
            if type(index) is not int or index < 0 or type(reading.axis) is not int or reading.axis not in (0, 1, 2):
                raise ValueError("observer endpoint identity")
            endpoints.append((frame, index, reading.axis))
        if len({e[0] for e in endpoints}) > 32:
            raise ValueError("observer frame ceiling")
        factors, row_owner, memberships = [], {}, 0
        for fid, ((u, v, relative), relation) in enumerate(sorted(registry.items())):
            memberships += sum(len(indices) for indices in relation.values())
            if memberships > 262144:
                raise ValueError("observer membership ceiling")
            indices = tuple(sorted({i for values in relation.values() for i in values}))
            for index in indices:
                if type(index) is not int or not 0 <= index < rows or index in row_owner:
                    raise ValueError("observer source row join")
                row_owner[index] = fid
            if id(relation) in self.relations:
                raise ValueError("observer relation alias")
            self.relations[id(relation)] = (relation, fid, u, v)
            factors.append((fid, u, v, relative, indices))
        if len(row_owner) != rows:
            raise ValueError("observer missing source row")
        self._emit("registry", {"source_digest": source_digest, "endpoints": tuple(endpoints),
            "factors": tuple(factors), "source_rows": rows, "registry_complete": True})
        self.registry_complete = True

    def _visited(self, index, q):
        if index >= self.MAX_ASSIGNMENTS:
            raise ValueError("observer assignment ceiling")
        self.observed["visited"] += 1
        self._emit("visited", {"assignment_index": index, "quadrants": tuple(q.items())})

    def _entered(self, index, q, factors):
        ids = []
        for u, v, relation in factors:
            stored = self.relations.get(id(relation))
            if stored is None or stored[0] is not relation or stored[2:] != (u, v):
                raise ValueError("observer foreign active relation")
            ids.append(stored[1])
        if len(set(ids)) != len(ids):
            raise ValueError("observer duplicate active relation")
        self.observed["entered"] += 1
        self._emit("entered", {"assignment_index": index, "quadrants": tuple(q.items()),
                              "active_factor_ids": tuple(ids)})

    def _returned(self, index, reason):
        self.observed["returned"] += 1
        self._emit("returned", {"assignment_index": index, "result_available": True, "reason": reason})

    def _classified(self, index):
        self.observed["classified"] += 1
        self._emit("classified", {"assignment_index": index})

    def _exception(self, stage, index=None, result_available=False, reason=None):
        self.complete = False
        payload = {"stage": stage, "assignment_index": index,
                   "result_available": result_available}
        if result_available:
            payload["reason"] = reason
        self._safe_emit("exception", payload)

    def finish(self, result):
        self._safe_emit("finished", {"status": result.status, "reason": result.reason,
            "source_digest": result.source_digest, "exhaustive": result.exhaustive,
            "observer_complete": self.complete, "registry_complete": self.registry_complete,
            "observed": tuple(self.observed.items()), "counters": tuple(result.counters.items()),
            "eligibility_reasons": tuple(result.eligibility_reasons.items())})

    def close(self):
        self.relations.clear()
        self.sink = None


def _propagate_partial_arcs(work, factors, incidence, domains, dirty, singleton):
    """Binary arc closure: canonical dirty factors, no relation/source pruning.

    Root starts every factor. A child starts only factors incident to its changed
    endpoint; its parent's domains are already arc-consistent. Each revision
    computes both supported endpoint sets from all original relation pairs.
    Remaining pairs support both endpoints, so the current factor is at a fixed
    point after simultaneous intersection; only other incident factors get dirty.
    Queue size is bounded by the existing registry-factor ceiling. Meter before
    queue mutation, incidence traversal, or relation-pair evaluation.
    """
    if any(not values for values in domains.values()):
        work.pruning["inconsistent_nodes"] += 1
        return False
    queue, queued = [], set()

    def enqueue(fid):
        if fid not in queued:
            work.relation_charge("arc_queue")
            queued.add(fid)
            heappush(queue, fid)

    if dirty is None:
        for fid in range(len(factors)):
            work.relation_charge("arc_incidence")
            enqueue(fid)
    else:
        for key in dirty:
            for fid in incidence[key]:
                work.relation_charge("arc_incidence")
                enqueue(fid)
    while queue:
        work.relation_charge("arc_queue")
        fid = heappop(queue)
        queued.remove(fid)
        work.counts["arc_factor_scans"] += 1
        u, v, relation = factors[fid]
        su, sv = set(), set()
        for a, b in sorted(relation):
            work.relation_charge("arc_relation")
            if a in domains[u] and b in domains[v]:
                su.add(a)
                sv.add(b)
        changed = []
        for key, supported in ((u, su), (v, sv)):
            removed = len(domains[key] - supported)
            if removed:
                work.pruning["values_removed"] += removed
                domains[key].intersection_update(supported)
                if len(domains[key]) == 1:
                    singleton(key)
                changed.append(key)
            if not domains[key]:
                work.pruning["inconsistent_nodes"] += 1
                return False
        for key in changed:
            for neighbor in incidence[key]:
                work.relation_charge("arc_incidence")
                if neighbor != fid:
                    enqueue(neighbor)
    return True


def solve_endpoint_hypotheses(readings, matches, *, session_frames, quadrant_constraints=(),
                              budget=EndpointBudget(), endpoint_incidence_sink=None,
                              _observability_mode="full"):
    """Classify canonical consistency with global precharged work ceilings.

    Every endpoint needs an active factor and every session frame must connect
    to the sorted production anchor on each world axis, before any solution is
    counted. A second distinct eligible (quadrants, alternatives) proves multiple;
    no/unique require exhaustive search. Resource refusal returns no choices,
    even if a candidate was found. Source digest binds input order; certificate
    row indices always address that exact original input, never a sorted copy.
    """
    if type(_observability_mode) is not str or _observability_mode not in ("full", "partial"):
        raise ValueError("invalid endpoint observability mode")
    validate_budget(budget, partial=_observability_mode == "partial")
    if (not isinstance(readings, Sequence) or isinstance(readings, (str, bytes))
            or not isinstance(matches, Sequence) or isinstance(matches, (str, bytes))
            or type(quadrant_constraints) is not tuple or type(session_frames) is not tuple):
        raise ValueError("readings/matches need sequences; session/constraints need immutable tuples")
    work, choices, source_digest = _Work(budget), [], None
    partial = None
    if _observability_mode == "partial":
        from . import partial_axis_observability as partial
        work.transcript_policy = partial.TRANSCRIPT_POLICY
        work.enable_partial_arcs()
    hooks = None if endpoint_incidence_sink is None else _EndpointIncidenceHooks(endpoint_incidence_sink)
    if hooks is not None: work.endpoint_incidence_hooks = hooks
    source = hashlib.sha256()
    exhaustive = False
    try:
        work.bound("endpoint_variables", len(readings))
        work.bound("input_match_rows", len(matches))
        work.bound("frames", len(session_frames))
        if any(type(frame) is not str or not frame for frame in session_frames):
            raise ValueError("session frames must be nonempty strings")
        work.session_frames = tuple(sorted(session_frames))
        source.update(work.encode(["universe", work.transcript_policy,
                                  plane_graph.PHYSICAL_DISPLACEMENT_POLICY,
                                  plane_graph.SAME_AXIS_TOLERANCE]))
        source.update(work.encode(["session_frames", session_frames]))
        for index, reading in enumerate(readings):
            if not isinstance(reading, PlaneReading):
                raise ValueError("non-reading input")
            work.bound("alternatives_per_endpoint", len(reading.alternatives))
            source.update(work.encode(["reading", index, reading.frame,
                                       reading_order_key(reading),
                                       tuple((i, float(a.blocks_per_period),
                                              float(a.distance_blocks), float(a.support))
                                             for i, a in enumerate(reading.alternatives))]))
        by_index, _, frames = plane_graph._canonical_reading_index(readings)
        if any(float(reading.side) not in (-1.0, 1.0) for reading in readings):
            raise ValueError("unverified physical reading side")
        work.bound("frames", len(frames))
        constraints = {}
        contradictory_constraints = False
        source.update(work.encode(["constraints", quadrant_constraints]))
        source.update(work.encode(["canonical-endpoints", tuple(sorted(by_index))]))
        for value in quadrant_constraints:
            if (type(value) is not tuple or len(value) != 3
                    or type(value[0]) is not str or type(value[1]) is not str
                    or value[0] == value[1] or value[0] not in frames or value[1] not in frames
                    or type(value[2]) is not int or value[2] not in range(4)):
                raise ValueError("malformed/foreign quadrant constraint")
            left, right, q = value
            if right < left: left, right, q = right, left, (-q) % 4
            key = (left, right)
            if key in constraints and constraints[key] != q:
                contradictory_constraints = True
            constraints[key] = q
        registry, evidence = {}, {}
        for row_index, row in enumerate(matches):
            left, right = plane_graph._validate_indexed_match(row, by_index)
            if type(row.quadrant_of_right) is not int or row.quadrant_of_right not in range(4):
                raise ValueError("invalid relative quadrant")
            if row.left == row.right:
                raise ValueError("matches must connect different frames")
            encoded = work.encode(["match", row_index, _row_payload(row)])
            source.update(encoded)
            work.charge("relation_visits", len(left.alternatives) + len(right.alternatives))
            selected_left = left.matching_alternatives(row.left_b)
            selected_right = right.matching_alternatives(row.right_b)
            if not selected_left or not selected_right:
                raise ValueError("unresolvable admitted physical alternative fails closed globally")
            il = tuple(i for i, a in enumerate(left.alternatives) if any(a is s for s in selected_left))
            ir = tuple(i for i, a in enumerate(right.alternatives) if any(a is s for s in selected_right))
            u, v = (row.left, row.left_plane), (row.right, row.right_plane)
            q = row.quadrant_of_right
            if v < u: u, v, il, ir, q = v, u, ir, il, (-q) % 4
            key = (u, v, q)
            if key not in registry:
                work.charge("registry_factors", 1)
                registry[key] = {}
            # Source evidence distinctions preserve ambiguity, not weight.
            cue = {name: value for name, value in _row_payload(row).items()
                   if name not in ("left", "right", "left_axis", "right_axis",
                                   "left_plane", "right_plane", "left_b", "right_b", "quadrant_of_right")}
            evidence[row_index] = hashlib.sha256(work.encode(cue)).hexdigest()
            for a in il:
                for b in ir:
                    work.charge("relation_visits", 1)
                    registry[key].setdefault((a, b), []).append(row_index)
        source_digest = source.hexdigest()
        work.bind_source(source_digest, tuple(sorted(by_index)))
        if hooks is not None: hooks.capture("registry", registry, by_index, len(matches), source_digest)
        _preflight_relations(work, by_index, matches, frames)
        if not frames or frames != work.session_frames:
            work.counts["planned_assignments"] = 0
            return _result("no", work, (), source_digest, True, "session_frame_mismatch")
        if contradictory_constraints:
            work.counts["planned_assignments"] = 0
            return _result("no", work, (), source_digest, True, "quadrant_constraint_inconsistent")
        partial_anchor = partial.select_anchor(work, by_index, frames) if partial is not None else None
        if partial is not None and partial_anchor is None:
            work.counts["planned_assignments"] = 0
            return _result("no", work, (), source_digest, True, "no_fully_observed_anchor")
        reason, count, plans = plane_graph._quadrant_assignment_plan(frames, constraints)
        work.counts["planned_assignments"] = count
        if reason == "component_too_large":
            # The shared planner reports zero rather than the over-cap count.
            work.counts["planned_assignments"] = None
            work.denial = ("quadrant_assignments", 0, plane_graph.MAX_ASSIGNMENTS + 1,
                           min(budget.quadrant_assignments, plane_graph.MAX_ASSIGNMENTS))
            raise _Exhausted
        if reason:
            return _result("no", work, (), source_digest, True, reason)
        work.bound("quadrant_assignments", count)
        activation_index = _factor_activation_index(work, registry)
        keys = tuple(sorted(by_index))
        for assignment_index, q in enumerate(plans):
            if partial is not None:
                # Preserve all relative assignments; fix the global orientation
                # gauge at a frame with all three measured local normals.
                offset = q[partial_anchor]
                q = {frame: (value - offset) % 4 for frame, value in q.items()}
            work.counts["visited_assignments"] += 1
            if hooks is not None: hooks.capture("visited", assignment_index, q)
            factors = _active_factors(work, activation_index, q)
            work.counts["active_factor_visits"] += len(factors)
            if hooks is not None: hooks.capture("entered", assignment_index, q, factors)
            try:
                if partial is None:
                    eligibility_reason = _eligibility_reason(work, factors, by_index, frames, q)
                else:
                    eligibility_reason, observation = partial.classify(
                        work, factors, by_index, frames, q, partial_anchor)
                    work.eligibility_digest.update(work.encode(
                        ["partial-axis", assignment_index, None if observation is None else observation.payload()],
                        category="disposition"))
            except BaseException:
                if hooks is not None: hooks.capture("exception", "eligibility", assignment_index)
                raise
            if hooks is not None: hooks.capture("returned", assignment_index, eligibility_reason)
            try:
                work.eligibility(assignment_index, q, eligibility_reason)
            except BaseException:
                if hooks is not None: hooks.capture("exception", "classification", assignment_index, True, eligibility_reason)
                raise
            if hooks is not None: hooks.capture("classified", assignment_index)
            if eligibility_reason is not None:
                continue
            domains = {key: set(range(len(by_index[key].alternatives))) for key in keys}
            forced = _ForcedDiameter(work, factors, by_index, matches, q)
            initial_pending = set()
            for key in keys:
                if len(domains[key]) == 1: forced.singleton(key, initial_pending)

            def walk(domains, pending, path=(), reserved=False, dirty=None):
                marker = len(forced.journal)
                try:
                    visit(domains, pending, path, reserved, dirty)
                finally:
                    # A failed undo aborts the invocation through the same atomic
                    # refusal path. No sibling ever observes partly restored state.
                    if work.denial is None: forced.undo(marker)

            def visit(domains, pending, path, reserved, dirty):
                if not reserved: work.charge("branch_nodes", 1)
                if partial is not None:
                    if not _propagate_partial_arcs(work, factors, forced.incidence,
                            domains, dirty, lambda key: forced.singleton(key, pending)):
                        return
                changed = partial is None
                while changed:
                    changed = False
                    if any(not values for values in domains.values()):
                        work.pruning["inconsistent_nodes"] += 1
                        return
                    for u, v, relation in factors:
                        su, sv = set(), set()
                        for a, b in sorted(relation):
                            work.charge("relation_visits", 1)
                            if a in domains[u] and b in domains[v]: su.add(a); sv.add(b)
                        for key, supported in ((u, su), (v, sv)):
                            removed = len(domains[key] - supported)
                            if removed:
                                work.pruning["values_removed"] += removed
                                domains[key].intersection_update(supported); changed = True
                                if len(domains[key]) == 1: forced.singleton(key, pending)
                            if not domains[key]:
                                work.pruning["inconsistent_nodes"] += 1
                                return
                if not forced.check(pending, domains, assignment_index, path): return
                free = [key for key in keys if len(domains[key]) > 1]
                if free:
                    branch = min(free, key=lambda key: (len(domains[key]), key))
                    for value in sorted(domains[branch]):
                        work.charge("branch_nodes", 1)  # before copying branch state
                        child = {key: set(values) for key, values in domains.items()}
                        child[branch] = {value}
                        child_pending = set()
                        forced.singleton(branch, child_pending)
                        walk(child, child_pending, path + ((forced.positions[branch], value),), True,
                             (branch,) if partial is not None else None)
                        if len(choices) == 2: return
                    return
                supports = []
                selected = {key: next(iter(domains[key])) for key in keys}
                for u, v, relation in factors:
                    rows = tuple(relation[(selected[u], selected[v])])
                    groups = {}
                    for i in rows: groups.setdefault(evidence[i], []).append(i)
                    supports.append(FactorSupport((u, v), (selected[u], selected[v]), rows,
                                                  tuple(tuple(g) for _, g in sorted(groups.items()))))
                work.counts["physical_assignments"] += 1
                physical_reason, physical_groups, physical_digest, witness = _physical_feasibility(
                    work, by_index, matches, q, selected, supports, frames, assignment_index)
                if physical_reason == "unverified_physical_displacement":
                    raise ValueError("unverified physical displacement fails closed globally")
                work.physical(assignment_index, q, selected, physical_reason, physical_digest, witness)
                if physical_reason is not None:
                    work.pruning["inconsistent_nodes"] += 1
                    return
                work.counts["solutions_found"] += 1
                work.charge("solution_rows", 1)  # only physically feasible assignments count
                # Bound the certificate's complete source-row support before retention.
                work.encode([assignment_index, tuple(q.items()), tuple(selected.items()),
                             [(s.endpoints, s.alternatives, s.source_rows,
                               s.distinguishable_support_groups) for s in supports]], category="support")
                choices.append(EndpointChoice(assignment_index, tuple(q.items()),
                               tuple((f, i, selected[(f, i)]) for f, i in keys), tuple(supports),
                               physical_groups, physical_digest))

            walk(domains, initial_pending)
            if len(choices) == 2:
                return _result("multiple", work, tuple(choices), source_digest, False)
        exhaustive = True
        if not choices:
            reason = ("no_eligible_assignment" if not work.counts["eligible_assignments"]
                      else "canonical_constraints_inconsistent")
            return _result("no", work, (), source_digest, exhaustive, reason)
        return _result("unique", work, tuple(choices), source_digest, exhaustive)
    except _Exhausted:
        return _result("budget_refused", work, (), source_digest, False)
    except BaseException:
        if hooks is not None: hooks.capture("exception", "solve")
        raise
    finally:
        if hooks is not None: hooks.close()


def _physical_feasibility(work, indexed, matches, quadrants, selected, supports, frames, assignment_index):
    """Stream relation extrema, then bounded deterministic difference constraints."""
    extrema, digest = {}, hashlib.sha256()
    digest.update(work.encode(["relations", work.transcript_policy, work.source_digest,
                              assignment_index,
                              tuple(selected[key] for key in work.endpoint_keys)], category="relation"))
    for factor in supports:
        for i in factor.source_rows:
            work.charge("relation_visits", 1)
            work.counts["physical_relation_visits"] += 1
            try:
                relation = plane_graph.signed_physical_relation(
                    indexed, matches[i], quadrants, selected, i, compact=True)
            except (ValueError, TypeError, KeyError, IndexError, OverflowError):
                return "unverified_physical_displacement", (), digest.hexdigest(), (i,)
            key, delta, row_id, left_alt, right_alt = relation
            reference = (row_id, left_alt, right_alt, key[2], delta)
            digest.update(work.encode(["relation", reference], category="relation"))
            if key not in extrema:
                extrema[key] = [delta, delta, reference, reference]
            else:
                group = extrema[key]
                if delta < group[0]: group[0], group[2] = delta, reference
                if delta > group[1]: group[1], group[3] = delta, reference
    groups = tuple((key, *value) for key, value in sorted(extrema.items()))
    for key, low, high, lo_witness, hi_witness in groups:
        work.charge("relation_visits", 1)
        if high - low > plane_graph.SAME_AXIS_TOLERANCE:
            return "physical_displacement_diameter", (), digest.hexdigest(), (lo_witness, hi_witness)
    for axis in range(3):
        distance, predecessor = dict.fromkeys(frames, 0.0), {}
        for iteration in range(len(frames)):
            changed = None
            for key, low, high, lo_witness, hi_witness in groups:
                f, g, world = key
                if world != axis: continue
                for u, v, bound in ((f, g, low + plane_graph.SAME_AXIS_TOLERANCE),
                                    (g, f, -(high - plane_graph.SAME_AXIS_TOLERANCE))):
                    work.charge("relation_visits", 1)
                    work.counts["physical_cycle_relaxations"] += 1
                    proposed = distance[u] + bound
                    if not math.isfinite(proposed):
                        raise ValueError("unverified physical cycle arithmetic fails closed globally")
                    if proposed < distance[v]:
                        distance[v] = proposed
                        predecessor[v] = (u, v, axis, bound, lo_witness[0], hi_witness[0])
                        changed = v
            if changed is None: break
            if iteration == len(frames) - 1:
                node = changed
                for _ in frames:
                    work.charge("relation_visits", 1)
                    node = predecessor[node][0]
                start, witness = node, []
                for _ in frames:
                    work.charge("relation_visits", 1)
                    edge = predecessor[node]
                    witness.append(edge)
                    node = edge[0]
                    if node == start: break
                references = tuple((frames.index(u), frames.index(v), axis, bound, lo, hi)
                                   for u, v, axis, bound, lo, hi in witness)
                return "physical_frame_cycle", (), digest.hexdigest(), references
    for group in groups:
        work.encode(["support-extrema", group], category="support")
    return None, groups, digest.hexdigest(), ()


def _eligibility_reason(work, factors, by_index, frames, quadrants):
    """Structural part of production active-constraint validation, no numerics."""
    if not factors:
        return "no_active_factors"
    supported = set()
    adjacency = {axis: {frame: set() for frame in frames} for axis in range(3)}
    rows_per_axis = {axis: 0 for axis in range(3)}
    for u, v, relation in factors:
        work.charge("relation_visits", 1)
        supported.update((u, v))
        left_axis, _ = plane_graph.axis_transform(by_index[u].axis, quadrants[u[0]])
        right_axis, _ = plane_graph.axis_transform(by_index[v].axis, quadrants[v[0]])
        if left_axis != right_axis:
            return "active_axis_mismatch"
        adjacency[left_axis][u[0]].add(v[0])
        adjacency[left_axis][v[0]].add(u[0])
        rows_per_axis[left_axis] += 1
    if supported != set(by_index):
        return "unsupported_reading_endpoint"
    anchor = frames[0]  # _canonical_reading_index sorts exactly like production.
    for axis in range(3):
        visited, queue = {anchor}, [anchor]
        while queue:
            for neighbour in sorted(adjacency[axis][queue.pop()]):
                work.charge("relation_visits", 1)
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        if not rows_per_axis[axis] or visited != set(frames):
            return f"axis_{axis}_disconnected"
    return None


def _result(status, work, choices, source_digest, exhaustive, reason=None):
    work.counts["eligibility_unclassified_assignments"] = (
        work.counts["visited_assignments"] - work.counts["eligibility_hashed_assignments"])
    planned = work.counts["planned_assignments"]
    work.counts["unvisited_assignments"] = (
        None if planned is None else planned - work.counts["visited_assignments"])
    ambiguous = any(len(f.distinguishable_support_groups) > 1
                    for choice in choices for f in choice.factors)
    if status == "unique" and ambiguous:
        status, reason, choices = "unresolved_evidence", "distinguishable_factor_support", ()
    result = EndpointResult(status, source_digest, choices,
                          MappingProxyType(dict(work.counts)),
                          MappingProxyType(dict(work.pruning)), work.denial,
                          exhaustive, ambiguous, reason, work.session_frames,
                          work.eligibility_digest.hexdigest(),
                          MappingProxyType(dict(work.eligibility_reasons)), tuple(work.exclusions),
                          plane_graph.PHYSICAL_DISPLACEMENT_POLICY, plane_graph.SAME_AXIS_TOLERANCE,
                          work.physical_digest.hexdigest(), tuple(work.physical_exclusions),
                          tuple(work.partial_exclusions),
                          MappingProxyType(dict(work.physical_reasons)), work.transcript_policy)
    hooks = getattr(work, "endpoint_incidence_hooks", None)
    if hooks is not None: hooks.finish(result)
    return result
