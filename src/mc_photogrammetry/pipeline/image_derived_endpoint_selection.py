"""Versioned full-candidate physical certificates; construction is not source replay.

Only live production derive or explicit replay in image_derived_joint_graph
re-establishes source-backed uniqueness. Validation recomputes the entire proof.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from types import SimpleNamespace

from mc_photogrammetry.pipeline import image_derived_graph as legacy
from mc_photogrammetry.pipeline import image_derived_signatures as signatures_mod
from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp
from mc_photogrammetry.reconstruction import lattice_planes, plane_graph
from mc_photogrammetry.reconstruction import partial_axis_observability as partial_axes

ENDPOINT_SELECTION_SCHEMA = "mc-photogrammetry-endpoint-selection-v4"
ENDPOINT_SELECTION_PRODUCER = "m9-g-endpoint-selection-4"
PARTIAL_SELECTION_SCHEMA = "mc-photogrammetry-partial-endpoint-selection-v1"
PARTIAL_SELECTION_PRODUCER = "m9-g-partial-endpoint-selection-1"
MAX_RECORD_BYTES = 16 * 1024 * 1024
MAX_PROOF_BYTES = 4 * 1024 * 1024
MAX_FRAME_BYTES = 4096
STATES = ("no", "unique", "multiple", "unresolved_evidence", "budget_refused")


class _RecordExhausted(ValueError):
    def __init__(self, name, limit):
        self.name, self.limit = name, limit
        super().__init__(f"endpoint {name} exceeds encoded bound")


def _encode(value, *, limit=None, name="record_bytes"):
    # Structural records have separately capped rows; bound strings and traversal
    # before allocating JSON chunks, then bound the final bytes before retention.
    limit = MAX_RECORD_BYTES if limit is None else limit
    stack, items, text = [(value, 0)], 0, 0
    while stack:
        item, depth = stack.pop()
        items += 1
        if depth > 32 or items + text > limit:
            raise _RecordExhausted(name, limit)
        if isinstance(item, (tuple, list, dict)):
            children = item.values() if isinstance(item, dict) else item
            if len(item) + len(stack) + items + text > limit:
                raise _RecordExhausted(name, limit)
            if isinstance(item, dict):
                text += sum(len(key) for key in item)
            stack.extend((child, depth + 1) for child in children)
        elif isinstance(item, str):
            text += len(item)
    data = bytearray()
    for part in json.JSONEncoder(sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).iterencode(value):
        if len(part) > limit - len(data):
            raise _RecordExhausted(name, limit)
        raw = part.encode("utf-8")
        if len(data) + len(raw) > limit:
            raise _RecordExhausted(name, limit)
        data.extend(raw)
    return bytes(data)


def _sha(value):
    return hashlib.sha256(_encode(value)).hexdigest()


def _hex(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _frames(frames):
    if (type(frames) is not tuple or not frames or len(frames) > csp.EndpointBudget().frames
            or len(set(frames)) != len(frames)
            or any(type(f) is not str or not f or len(f.encode("utf-8")) > MAX_FRAME_BYTES for f in frames)):
        raise ValueError("endpoint frames are malformed or exceed bound")


def candidate_digest(matches):
    if type(matches) is not tuple or len(matches) > csp.EndpointBudget().input_match_rows:
        raise ValueError("candidate rows must be a bounded complete tuple")
    if any(type(row) is not plane_graph.PlaneMatch for row in matches):
        raise ValueError("candidate row type is foreign")
    digest, used = hashlib.sha256(b"["), 2
    for i, row in enumerate(matches):
        raw = _encode(legacy._match_payload(row), limit=csp.EndpointBudget().scalar_row_bytes)
        used += len(raw) + bool(i)
        if used > MAX_RECORD_BYTES:
            raise _RecordExhausted("record_bytes", MAX_RECORD_BYTES)
        if i: digest.update(b",")
        digest.update(raw)
    digest.update(b"]")
    return digest.hexdigest()


def reading_bindings(readings, evidence):
    if type(readings) is not tuple or len(readings) > csp.EndpointBudget().endpoint_variables:
        raise ValueError("reading universe exceeds endpoint bound")
    if type(evidence) is not tuple or len(evidence) > 128 * 16:
        raise ValueError("signature universe exceeds endpoint bound")
    # Counts/types precede canonical-key traversal and expected-universe lists.
    for reading in readings:
        if (type(reading) is not lattice_planes.PlaneReading or type(reading.alternatives) is not tuple
                or len(reading.alternatives) > csp.EndpointBudget().alternatives_per_endpoint):
            raise ValueError("reading alternative count/type exceeds bound")
    if any(type(item) is not signatures_mod.SignatureEvidence for item in evidence):
        raise ValueError("foreign signature evidence")
    canonical, identities, _ = plane_graph._canonical_reading_index(readings)
    expected = [(i, a) for i, r in enumerate(readings) for a in range(len(r.alternatives))]
    if [(e.canonical_index, e.alternative_index) for e in evidence] != expected:
        raise ValueError("signature evidence does not cover the complete ordered universe")
    sigs = {}
    for item in evidence:
        if type(item) is not signatures_mod.SignatureEvidence:
            raise ValueError("foreign signature evidence")
        signatures_mod.SignatureEvidence(**legacy._fields(item))
        r = readings[item.canonical_index]
        a = r.alternatives[item.alternative_index]
        if (item.signature.reading is not r or item.frame_id != r.frame
                or item.reading_axis != r.axis or item.blocks_per_period != a.blocks_per_period):
            raise ValueError("signature evidence is foreign to the reading alternative")
        sigs[(item.canonical_index, item.alternative_index)] = item.signature_sha256
    rows = []
    for global_index, r in enumerate(readings):
        if len(r.alternatives) > csp.EndpointBudget().alternatives_per_endpoint:
            raise ValueError("reading alternative count exceeds bound")
        key = identities[id(r)]
        alternatives = tuple((float(a.blocks_per_period), float(a.distance_blocks),
                              float(a.support)) for a in r.alternatives)
        rows.append((key, global_index, r.axis, _sha(lattice_planes.reading_order_key(r)),
                     alternatives, tuple(sigs[(global_index, a)] for a in range(len(alternatives)))))
    assert len(canonical) == len(rows)
    return tuple(sorted(rows))


def _proof_record(result):
    return {field.name: (dict(value) if field.name in ("counters", "pruning", "eligibility_reasons", "physical_reasons")
                        else [dataclasses.asdict(choice) for choice in value] if field.name == "choices"
                        else value)
            for field in dataclasses.fields(csp.EndpointResult)
            for value in (getattr(result, field.name),)}


def _proof_json(result):
    raw = _encode(_proof_record(result), limit=MAX_PROOF_BYTES, name="proof_bytes")
    return raw.decode("utf-8")


def _byte_refusal(result, error):
    counts = dict(result.counters)
    counts["omitted_exclusions"] += counts["retained_exclusions"]
    counts["retained_exclusions"] = 0
    counts["omitted_physical_exclusions"] += counts["retained_physical_exclusions"]
    counts["retained_physical_exclusions"] = 0
    counts["omitted_partial_exclusions"] += counts["retained_partial_exclusions"]
    counts["retained_partial_exclusions"] = 0
    return dataclasses.replace(result, status="budget_refused", choices=(), exhaustive=False,
        reason="endpoint_budget_refused", counters=counts, excluded_assignments=(),
        physical_exclusions=(), partial_exclusions=(),
        denial=(error.name, 0, error.limit + 1, error.limit))


def _bounded_result(result):
    try:
        return result, _proof_json(result)
    except _RecordExhausted as error:
        refused = _byte_refusal(result, error)
        return refused, _proof_json(refused)


def _validate_relation_reference(ref, count):
    if (type(ref) is not list or len(ref) != 5
            or type(ref[0]) is not int or not 0 <= ref[0] < count
            or any(type(a) is not int or not 0 <= a < 16 for a in ref[1:3])
            or type(ref[3]) is not int or ref[3] not in range(3)
            or type(ref[4]) is not float or not math.isfinite(ref[4])):
        raise ValueError("endpoint compact relation reference differs")


def _validate_proof(proof, status, bindings, frame_ids, count, budget, *,
                    _transcript_policy=csp.TRANSCRIPT_POLICY, _quadrant_anchor=None):
    names = {f.name for f in dataclasses.fields(csp.EndpointResult)}
    if type(proof) is not dict or set(proof) != names or proof["status"] != status:
        raise ValueError("endpoint proof fields/status differ")
    if proof["transcript_policy"] != _transcript_policy:
        raise ValueError("unsupported endpoint transcript policy")
    counters, pruning = proof["counters"], proof["pruning"]
    expected_counts = set(csp._Work(budget).counts) | {"eligibility_unclassified_assignments", "unvisited_assignments"}
    if _transcript_policy == partial_axes.TRANSCRIPT_POLICY:
        expected_counts.update(csp.PARTIAL_ARC_COUNTERS)
    if (type(counters) is not dict or set(counters) != expected_counts
            or any((value is None and name not in ("planned_assignments", "unvisited_assignments"))
                   or (value is not None and (type(value) is not int or value < 0))
                   for name, value in counters.items())
            or type(pruning) is not dict or set(pruning) != {"values_removed", "inconsistent_nodes"}
            or any(type(v) is not int or v < 0 for v in pruning.values())):
        raise ValueError("endpoint counters are malformed")
    for field in dataclasses.fields(budget):
        if counters[field.name] > getattr(budget, field.name):
            raise ValueError("endpoint counter exceeds exact budget")
    if (sum(counters[category + "_bytes"] for category in csp.BYTE_CATEGORIES)
            != counters["scalar_payload_bytes"]
            or counters["scalar_records_attempted"] != counters["scalar_records_committed"]
               + counters["scalar_records_uncommitted"]
            or (status != "budget_refused" and counters["scalar_records_uncommitted"])):
        raise ValueError("endpoint scalar byte accounting does not close")
    visited, hashed = counters["visited_assignments"], counters["eligibility_hashed_assignments"]
    if (counters["eligible_assignments"] + counters["excluded_assignments"] != hashed
            or hashed + counters["eligibility_unclassified_assignments"] != visited
            or counters["retained_exclusions"] + counters["omitted_exclusions"] != counters["excluded_assignments"]
            or counters["retained_exclusions"] > 128):
        raise ValueError("endpoint accounting does not close")
    planned = counters["planned_assignments"]
    if planned is not None and (planned > 16384 or counters["unvisited_assignments"] is None
                               or visited + counters["unvisited_assignments"] != planned):
        raise ValueError("endpoint assignment accounting does not close")
    if (proof["session_frames"] != list(sorted(frame_ids))
            and not (status == "budget_refused" and proof["source_digest"] is None and proof["session_frames"] == [])):
        raise ValueError("endpoint proof changes session frames")
    if type(proof["exhaustive"]) is not bool or type(proof["evidence_ambiguous"]) is not bool:
        raise ValueError("endpoint proof flags must be bool")
    if not _hex(proof["eligibility_digest"]) or (proof["source_digest"] is not None and not _hex(proof["source_digest"])):
        raise ValueError("endpoint proof digest is malformed")
    reasons, exclusions = proof["eligibility_reasons"], proof["excluded_assignments"]
    if (type(reasons) is not dict or any(type(k) is not str or not k or type(v) is not int or v <= 0
                                       for k, v in reasons.items())
            or sum(reasons.values()) != counters["excluded_assignments"]
            or type(exclusions) is not list or len(exclusions) != counters["retained_exclusions"]):
        raise ValueError("endpoint exclusion records are incomplete")
    for i, q, reason in exclusions:
        if (type(i) is not int or not 0 <= i < visited or reason not in reasons
                or type(q) is not list or [p[0] for p in q] != sorted(frame_ids)
                or any(type(p[1]) is not int or p[1] not in range(4) for p in q)):
            raise ValueError("endpoint exclusion row is malformed")
    if [row[0] for row in exclusions] != sorted(set(row[0] for row in exclusions)):
        raise ValueError("endpoint exclusion order/identity differs")
    choices = proof["choices"]
    n = counters["solutions_found"]
    if (type(choices) is not list or len(choices) > 2
            or (status == "unique" and (n != 1 or len(choices) != 1 or not proof["exhaustive"] or proof["evidence_ambiguous"]))
            or (status == "multiple" and (n != 2 or len(choices) != 2 or proof["exhaustive"]))
            or (status == "unresolved_evidence" and (n != 1 or choices or not proof["exhaustive"] or not proof["evidence_ambiguous"]))
            or (status == "no" and (n != 0 or choices or not proof["exhaustive"]))
            or (status == "budget_refused" and (choices or proof["exhaustive"]))):
        raise ValueError("endpoint disposition and canonical counts differ")
    if proof["exhaustive"] and (counters["unvisited_assignments"] or counters["eligibility_unclassified_assignments"]):
        raise ValueError("exhaustive endpoint proof has unvisited work")
    if status != "budget_refused" and (not _hex(proof["source_digest"])
            or counters["endpoint_variables"] != len(bindings) or counters["input_match_rows"] != count
            or counters["frames"] != len(frame_ids) or counters["solution_rows"] != n):
        raise ValueError("endpoint source/candidate counters differ")
    denial = proof["denial"]
    if status == "budget_refused":
        caps = {f.name: getattr(budget, f.name) for f in dataclasses.fields(budget)}
        caps.update(proof_bytes=MAX_PROOF_BYTES, record_bytes=MAX_RECORD_BYTES)
        if (type(denial) is not list or len(denial) != 4 or denial[0] not in caps
                or any(type(v) is not int or v < 0 for v in denial[1:])
                or denial[3] != caps[denial[0]] or denial[1] + denial[2] <= denial[3]):
            raise ValueError("endpoint resource denial does not replay structurally")
    elif denial is not None:
        raise ValueError("non-refused endpoint proof carries resource denial")
    if (proof["physical_policy"] != plane_graph.PHYSICAL_DISPLACEMENT_POLICY
            or proof["physical_tolerance"] != plane_graph.SAME_AXIS_TOLERANCE
            or not _hex(proof["physical_digest"])
            or counters["physical_hashed_assignments"] != counters["physical_feasible"] + counters["physical_rejected"]
            or counters["physical_hashed_assignments"] > counters["physical_assignments"]
            or counters["physical_assignments"] > counters["branch_nodes"]
            or counters["physical_feasible"] < n
            or counters["retained_physical_exclusions"] + counters["omitted_physical_exclusions"] != counters["physical_rejected"]
            or counters["retained_physical_exclusions"] + counters["retained_partial_exclusions"] > 128
            or type(proof["physical_exclusions"]) is not list
            or len(proof["physical_exclusions"]) != counters["retained_physical_exclusions"]
            or type(proof["physical_reasons"]) is not dict
            or any(type(k) is not str or type(v) is not int or v <= 0 for k, v in proof["physical_reasons"].items())
            or sum(proof["physical_reasons"].values()) != counters["physical_rejected"]):
        raise ValueError("endpoint physical policy/accounting differs")
    if (counters["retained_partial_exclusions"] + counters["omitted_partial_exclusions"] != counters["partial_prunes"]
            or counters["partial_prunes"] > counters["partial_checks"]
            or counters["search_events_attempted"] != counters["search_events_committed"] + counters["search_events_uncommitted"]
            or counters["search_events_committed"] != counters["physical_hashed_assignments"] + counters["partial_prunes"]
            or (status != "budget_refused" and counters["search_events_uncommitted"])
            or type(proof["partial_exclusions"]) is not list
            or len(proof["partial_exclusions"]) != counters["retained_partial_exclusions"]
            or sum(counters[name + "_visits"] for name in (
                "preflight", "incidence", "partial_support", "partial_update", "partial_queue",
                "partial_journal", "partial_undo", "partial_event")) > counters["relation_visits"]):
        raise ValueError("endpoint partial event accounting does not close")
    if _transcript_policy == partial_axes.TRANSCRIPT_POLICY:
        arc_visits = sum(counters[name] for name in csp.PARTIAL_ARC_COUNTERS if name != "arc_factor_scans")
        other_visits = sum(counters[name + "_visits"] for name in (
            "preflight", "incidence", "partial_support", "partial_update", "partial_queue",
            "partial_journal", "partial_undo", "partial_event"))
        if (arc_visits + other_visits + counters["physical_relation_visits"]
                + counters["physical_cycle_relaxations"] > counters["relation_visits"]
                or counters["arc_factor_scans"] > counters["arc_queue_visits"]):
            raise ValueError("endpoint partial arc accounting does not close")
    endpoint_alts = {tuple(row[0]): row[4] for row in bindings}
    alt_sizes = [len(endpoint_alts[key]) for key in sorted(endpoint_alts)]
    for event in proof["partial_exclusions"]:
        if type(event) is not list or len(event) != 6 or event[0] != "partial":
            raise ValueError("endpoint partial event kind/fields differ")
        _, index, path, masks, forced, group = event
        if (type(index) is not int or not 0 <= index < visited
                or type(path) is not list or len(path) > len(alt_sizes)
                or type(masks) is not list or len(masks) != len(alt_sizes)
                or any(type(mask) is not int or not 0 < mask < (1 << size)
                       for mask, size in zip(masks, alt_sizes))):
            raise ValueError("endpoint partial frontier differs")
        decisions = set()
        for decision in path:
            if (type(decision) is not list or len(decision) != 2
                    or type(decision[0]) is not int or not 0 <= decision[0] < len(alt_sizes)
                    or type(decision[1]) is not int or not 0 <= decision[1] < alt_sizes[decision[0]]
                    or decision[0] in decisions or masks[decision[0]] != 1 << decision[1]):
                raise ValueError("endpoint partial branch decisions differ")
            decisions.add(decision[0])
        if type(forced) is not list or not forced or len(forced) > counters["registry_factors"]:
            raise ValueError("endpoint partial forced support is absent")
        factor_ids, supported_rows = [], set()
        for factor in forced:
            if type(factor) is not list or len(factor) != 6:
                raise ValueError("endpoint partial factor fields differ")
            fid, u, a, v, b, rows = factor
            if (any(type(value) is not int for value in (fid, u, a, v, b))
                    or not 0 <= fid < counters["registry_factors"]
                    or not 0 <= u < v < len(alt_sizes)
                    or not 0 <= a < alt_sizes[u] or not 0 <= b < alt_sizes[v]
                    or masks[u] != 1 << a or masks[v] != 1 << b
                    or type(rows) is not list or not rows or rows != sorted(set(rows))
                    or any(type(row) is not int or not 0 <= row < count for row in rows)):
                raise ValueError("endpoint partial factor identity differs")
            factor_ids.append(fid); supported_rows.update(rows)
        if factor_ids != sorted(set(factor_ids)):
            raise ValueError("endpoint partial factor order differs")
        if type(group) is not list or len(group) != 5:
            raise ValueError("endpoint partial extrema fields differ")
        key, low, high, lo, hi = group
        if (type(key) is not list or len(key) != 3 or key[0] not in frame_ids
                or key[1] not in frame_ids or key[0] >= key[1]
                or type(key[2]) is not int or key[2] not in range(3)
                or type(low) is not float or type(high) is not float
                or not math.isfinite(low) or not math.isfinite(high)
                or high - low <= plane_graph.SAME_AXIS_TOLERANCE):
            raise ValueError("endpoint partial diameter differs")
        for ref, delta in ((lo, low), (hi, high)):
            _validate_relation_reference(ref, count)
            if ref[0] not in supported_rows or ref[3] != key[2] or ref[4] != delta:
                raise ValueError("endpoint partial extrema witness differs")
    for exclusion in proof["physical_exclusions"]:
        if type(exclusion) is not list or len(exclusion) != 5:
            raise ValueError("endpoint compact exclusion fields differ")
        index, vector, code, digest, witness = exclusion
        if (type(index) is not int or not 0 <= index < visited
                or type(vector) is not list or len(vector) != len(alt_sizes)
                or any(type(a) is not int or not 0 <= a < n for a, n in zip(vector, alt_sizes))
                or type(code) is not int or code not in (1, 2) or not _hex(digest)
                or type(witness) is not list or not witness):
            raise ValueError("endpoint compact exclusion identity differs")
        if code == 1:
            if len(witness) != 2:
                raise ValueError("endpoint diameter witnesses are incomplete")
            for ref in witness:
                _validate_relation_reference(ref, count)
        else:
            for edge in witness:
                if (type(edge) is not list or len(edge) != 6
                        or any(type(v) is not int or not 0 <= v < len(frame_ids) for v in edge[:2])
                        or type(edge[2]) is not int or edge[2] not in range(3)
                        or type(edge[3]) is not float or not math.isfinite(edge[3])
                        or any(type(v) is not int or not 0 <= v < count for v in edge[4:])):
                    raise ValueError("endpoint compact cycle witness differs")
    seen = set()
    ambiguous = False
    for choice in choices:
        if type(choice) is not dict or set(choice) != {"assignment_index", "quadrants", "alternatives", "factors", "physical_groups", "physical_relations_digest"}:
            raise ValueError("endpoint choice fields differ")
        if not _hex(choice["physical_relations_digest"]) or type(choice["physical_groups"]) is not list or not choice["physical_groups"]:
            raise ValueError("endpoint physical relations are absent")
        for group in choice["physical_groups"]:
            if type(group) is not list or len(group) != 5:
                raise ValueError("endpoint compact extrema fields differ")
            _validate_relation_reference(group[3], count)
            _validate_relation_reference(group[4], count)
        q, alts = choice["quadrants"], choice["alternatives"]
        if ([p[0] for p in q] != sorted(frame_ids)
                or dict(q).get(_quadrant_anchor or sorted(frame_ids)[0]) != 0
                or any(type(p[1]) is not int or p[1] not in range(4) for p in q)
                or type(choice["assignment_index"]) is not int or not 0 <= choice["assignment_index"] < visited):
            raise ValueError("endpoint choice quadrant identity differs")
        selected = {(f, i): a for f, i, a in alts}
        if (len(selected) != len(alts) or list(selected) != sorted(endpoint_alts)
                or any(type(a) is not int or not 0 <= a < len(endpoint_alts[k]) for k, a in selected.items())):
            raise ValueError("endpoint choice alternative identity differs")
        identity = _sha([q, alts])
        if identity in seen:
            raise ValueError("duplicate canonical choice multiplies disposition")
        seen.add(identity)
        factors, pairs, supported = choice["factors"], [], set()
        if not factors:
            raise ValueError("endpoint choice has no mandatory support")
        for factor in factors:
            if set(factor) != {"endpoints", "alternatives", "source_rows", "distinguishable_support_groups"}:
                raise ValueError("endpoint factor fields differ")
            u, v = map(tuple, factor["endpoints"])
            pair, source_rows, groups = factor["alternatives"], factor["source_rows"], factor["distinguishable_support_groups"]
            if (u >= v or u not in selected or v not in selected or pair != [selected[u], selected[v]]
                    or not source_rows or source_rows != sorted(set(source_rows))
                    or any(type(i) is not int or not 0 <= i < count for i in source_rows)
                    or not groups or any(not g or g != sorted(set(g)) for g in groups)
                    or sorted(i for g in groups for i in g) != source_rows):
                raise ValueError("endpoint factor support is malformed")
            ambiguous |= len(groups) > 1
            pairs.append((u, v)); supported.update((u, v))
        if pairs != sorted(set(pairs)) or supported != set(endpoint_alts):
            raise ValueError("endpoint factor coverage/order differs")
    if choices and ambiguous != proof["evidence_ambiguous"]:
        raise ValueError("endpoint evidence ambiguity flag differs")


@dataclass(frozen=True, slots=True)
class RepresentativeConstraint:
    factor_index: int
    source_row: int
    match: plane_graph.PlaneMatch


@dataclass(frozen=True, slots=True)
class EndpointSelectionCertificate:
    schema: str
    producer: str
    status: str
    reason: str
    session_sha256: str
    lattice_sha256: str
    signatures_sha256: str
    connection_sha256: str
    frame_ids: tuple[str, ...]
    quadrant_constraints: tuple[tuple[str, str, int], ...]
    budget: csp.EndpointBudget
    reading_bindings: tuple
    candidate_count: int
    candidate_matches_sha256: str
    proof_json: str
    representatives: tuple[RepresentativeConstraint, ...]
    candidate_dispositions: tuple = ()
    result_sha256: str = ""

    def __post_init__(self):
        partial = type(self) is PartialEndpointSelectionCertificate
        schema = PARTIAL_SELECTION_SCHEMA if partial else ENDPOINT_SELECTION_SCHEMA
        producer = PARTIAL_SELECTION_PRODUCER if partial else ENDPOINT_SELECTION_PRODUCER
        if self.schema != schema or self.producer != producer or self.status not in STATES:
            raise ValueError("unsupported endpoint selection identity/status")
        _frames(self.frame_ids)
        if (type(self.reason) is not str or not self.reason
                or any(not _hex(v) for v in (self.session_sha256, self.lattice_sha256, self.signatures_sha256,
                                            self.candidate_matches_sha256))
                or self.connection_sha256 != "" and not _hex(self.connection_sha256)):
            raise ValueError("endpoint selection bindings are malformed")
        csp.validate_budget(self.budget, partial=partial)
        if type(self.quadrant_constraints) is not tuple:
            raise ValueError("endpoint constraints must be an immutable tuple")
        for row in self.quadrant_constraints:
            if (type(row) is not tuple or len(row) != 3 or row[0] == row[1]
                    or row[0] not in self.frame_ids or row[1] not in self.frame_ids
                    or type(row[2]) is not int or row[2] not in range(4)):
                raise ValueError("endpoint quadrant constraint is malformed")
        if type(self.reading_bindings) is not tuple or len(self.reading_bindings) > 128:
            raise ValueError("endpoint reading bindings exceed bound")
        keys, global_indices = [], []
        for row in self.reading_bindings:
            if type(row) is not tuple or len(row) != 6:
                raise ValueError("endpoint reading binding shape differs")
            endpoint, index, axis, digest, alternatives, sigs = row
            if (type(endpoint) is not tuple or len(endpoint) != 2 or endpoint[0] not in self.frame_ids
                    or type(endpoint[1]) is not int or endpoint[1] < 0
                    or type(index) is not int or index < 0 or type(axis) is not int or axis not in range(3)
                    or not _hex(digest) or type(alternatives) is not tuple or len(alternatives) > 16
                    or type(sigs) is not tuple or len(sigs) != len(alternatives) or any(not _hex(s) for s in sigs)
                    or any(type(a) is not tuple or len(a) != 3 or any(type(v) is not float for v in a)
                           for a in alternatives)):
                raise ValueError("endpoint reading binding is malformed")
            keys.append(endpoint); global_indices.append(index)
        if keys != sorted(set(keys)) or sorted(global_indices) != list(range(len(keys))):
            raise ValueError("endpoint reading bindings are not complete/order-stable")
        for frame in self.frame_ids:
            indices = [i for f, i in keys if f == frame]
            if indices != list(range(len(indices))):
                raise ValueError("endpoint per-frame indices are not canonical")
        if type(self.candidate_count) is not int or not 0 <= self.candidate_count <= 16384:
            raise ValueError("endpoint candidate count exceeds bound")
        if (type(self.candidate_dispositions) is not tuple
                or len(self.candidate_dispositions) != self.candidate_count
                or any(d not in ("admitted", plane_graph.SAME_AXIS_DISPOSITION_KEPT,
                                 plane_graph.SAME_AXIS_DISPOSITION_DROPPED) for d in self.candidate_dispositions)):
            raise ValueError("endpoint admitted universe dispositions differ")
        if type(self.proof_json) is not str or len(self.proof_json.encode("utf-8")) > MAX_PROOF_BYTES:
            raise ValueError("endpoint proof is not a bounded scalar record")
        try:
            proof = json.loads(self.proof_json)
        except (ValueError, RecursionError) as error:
            raise ValueError("endpoint proof does not decode") from error
        if _encode(proof).decode("utf-8") != self.proof_json:
            raise ValueError("endpoint proof is not canonical")
        proof_options = {}
        if partial:
            anchor = _partial_anchor(self.reading_bindings, self.frame_ids)
            proof_options = dict(_transcript_policy=partial_axes.TRANSCRIPT_POLICY,
                                 _quadrant_anchor=anchor)
        _validate_proof(proof, self.status, self.reading_bindings, self.frame_ids, self.candidate_count, self.budget,
                        **proof_options)
        if partial:
            expected_observability = _partial_observability(proof, self.reading_bindings, self.frame_ids, self.budget)
            if (type(self.observability) is not tuple or self.observability != expected_observability
                    or any(type(value) is not partial_axes.AxisObservability for value in self.observability)):
                raise ValueError("partial observability does not replay from mandatory factors")
        expected_reason = (("partial_geometry_not_evaluated" if partial else "joint_geometry_not_evaluated") if self.status == "unique" else
                           "multiple_endpoint_hypotheses" if self.status == "multiple" else
                           "endpoint_budget_refused" if self.status == "budget_refused" else proof["reason"])
        if self.reason != expected_reason:
            raise ValueError("endpoint terminal reason differs")
        if type(self.representatives) is not tuple or len(self.representatives) > 4096:
            raise ValueError("endpoint representative constraints exceed bound")
        if self.status != "unique" and self.representatives:
            raise ValueError("abstaining endpoint selection carries representatives")
        if self.status == "unique":
            factors = proof["choices"][0]["factors"]
            if len(self.representatives) != len(factors):
                raise ValueError("endpoint representative coverage differs")
            altmap = {row[0]: row[4] for row in self.reading_bindings}
            for i, (rep, factor) in enumerate(zip(self.representatives, factors)):
                if (type(rep) is not RepresentativeConstraint or type(rep.factor_index) is not int
                        or rep.factor_index != i or type(rep.source_row) is not int
                        or rep.source_row != factor["source_rows"][0] or type(rep.match) is not plane_graph.PlaneMatch):
                    raise ValueError("endpoint representative source/order differs")
                u, v = (rep.match.left, rep.match.left_plane), (rep.match.right, rep.match.right_plane)
                if sorted((u, v)) != [tuple(p) for p in factor["endpoints"]]:
                    raise ValueError("endpoint representative identity differs")
                selected = dict(zip(map(tuple, factor["endpoints"]), factor["alternatives"]))
                if (rep.match.left_b != altmap[u][selected[u]][0]
                        or rep.match.right_b != altmap[v][selected[v]][0]):
                    raise ValueError("endpoint representative b is not canonical")
        digest = _sha(_payload(self))
        if self.result_sha256 not in ("", digest):
            raise ValueError("endpoint selection digest mismatch")
        object.__setattr__(self, "result_sha256", digest)

    def proof_record(self):
        return json.loads(self.proof_json)


def _payload(cert):
    payload = {f.name: (dataclasses.asdict(cert.budget) if f.name == "budget" else
                     [[r.factor_index, r.source_row, legacy._match_payload(r.match)] for r in cert.representatives]
                     if f.name == "representatives" else
                     [value.payload() for value in cert.observability] if f.name == "observability" else getattr(cert, f.name))
            for f in dataclasses.fields(type(cert)) if f.name != "result_sha256"}
    if type(cert.budget) is csp.PartialEndpointBudget:
        payload["budget_profile"] = "partial-relations-524288-v1"
    return payload


@dataclass(frozen=True, slots=True)
class PartialEndpointSelectionCertificate(EndpointSelectionCertificate):
    """Selection authority only; unknown world offsets have no numeric value.

    This distinct type/schema is intentionally rejected by existing joint-graph
    consumers. Stage 2 must explicitly project only resolved geometry.
    """
    observability: tuple[partial_axes.AxisObservability, ...] = ()


def _partial_anchor(bindings, frame_ids):
    return next((frame for frame in sorted(frame_ids)
                 if {row[2] for row in bindings if row[0][0] == frame} == {0, 1, 2}), None)


def _partial_observability(proof, bindings, frame_ids, budget):
    work = _ValidationWork(budget)
    indexed = {row[0]: SimpleNamespace(axis=row[2]) for row in bindings}
    result = []
    for choice in proof["choices"]:
        factors = [(tuple(row["endpoints"][0]), tuple(row["endpoints"][1]), {})
                   for row in choice["factors"]]
        reason, observation = partial_axes.classify(work, factors, indexed, tuple(sorted(frame_ids)),
            dict(choice["quadrants"]), _partial_anchor(bindings, frame_ids))
        if reason is not None:
            raise ValueError("selected partial assignment is structurally ineligible")
        # Validate immutable shape and the complete frame partition, not a
        # caller-supplied boolean or a union of overlapping axis marginals.
        partial_axes.from_payload(observation.payload())
        result.append(observation)
    return tuple(result)


def select_endpoint_hypotheses(*, readings, signatures, candidate_matches, frame_ids,
                              quadrant_constraints, budget, session_sha256,
                              lattice_sha256, signatures_sha256, connection_sha256="",
                              candidate_dispositions=None, endpoint_incidence_sink=None,
                              _observability_mode="full"):
    """Select isolated canonical hypotheses; does not prove source authenticity."""
    csp.validate_budget(budget, partial=_observability_mode == "partial")
    bindings = reading_bindings(readings, signatures)
    digest = candidate_digest(candidate_matches)
    canonical, _, _ = plane_graph._canonical_reading_index(readings)
    for row in candidate_matches:
        plane_graph._validate_indexed_match(row, canonical)
    observer_options = ({} if endpoint_incidence_sink is None else
                        {"endpoint_incidence_sink": endpoint_incidence_sink})
    if _observability_mode != "full":
        observer_options["_observability_mode"] = _observability_mode
    result = csp.solve_endpoint_hypotheses(readings, candidate_matches, session_frames=frame_ids,
                quadrant_constraints=quadrant_constraints, budget=budget, **observer_options)
    dispositions = (("admitted",) * len(candidate_matches)
                    if candidate_dispositions is None else candidate_dispositions)
    return _finish_selection(result, canonical, candidate_matches, bindings, digest, dispositions,
        frame_ids=frame_ids, quadrant_constraints=quadrant_constraints, budget=budget,
        session_sha256=session_sha256, lattice_sha256=lattice_sha256,
        signatures_sha256=signatures_sha256, connection_sha256=connection_sha256,
        _partial_mode=_observability_mode == "partial")


def select_partial_endpoint_hypotheses(**kwargs):
    """Explicit offline partial-selection seam; ordinary callers remain full."""
    return select_endpoint_hypotheses(**kwargs, _observability_mode="partial")


def _finish_selection(result, canonical, candidate_matches, bindings, digest, dispositions, *,
                      frame_ids, quadrant_constraints, budget, session_sha256, lattice_sha256,
                      signatures_sha256, connection_sha256, _partial_mode=False):
    result, proof_json = _bounded_result(result)
    representatives = []
    if result.status == "unique":
        choice = result.choices[0]
        selected = {(f, i): a for f, i, a in choice.alternatives}
        for i, factor in enumerate(choice.factors):
            source_row = factor.source_rows[0]
            match = candidate_matches[source_row]
            left, right = (match.left, match.left_plane), (match.right, match.right_plane)
            representatives.append(RepresentativeConstraint(i, source_row, dataclasses.replace(
                match, left_b=canonical[left].alternatives[selected[left]].blocks_per_period,
                right_b=canonical[right].alternatives[selected[right]].blocks_per_period)))
    def construct(result, encoded, representatives):
        reason = (("partial_geometry_not_evaluated" if _partial_mode else "joint_geometry_not_evaluated") if result.status == "unique" else
                  "multiple_endpoint_hypotheses" if result.status == "multiple" else
                  "endpoint_budget_refused" if result.status == "budget_refused" else result.reason)
        cls = PartialEndpointSelectionCertificate if _partial_mode else EndpointSelectionCertificate
        options = (dict(observability=_partial_observability(json.loads(encoded), bindings, frame_ids, budget))
                   if _partial_mode else {})
        return cls(PARTIAL_SELECTION_SCHEMA if _partial_mode else ENDPOINT_SELECTION_SCHEMA,
            PARTIAL_SELECTION_PRODUCER if _partial_mode else ENDPOINT_SELECTION_PRODUCER,
            result.status, reason, session_sha256, lattice_sha256, signatures_sha256, connection_sha256,
            frame_ids, quadrant_constraints, budget, bindings, len(candidate_matches), digest,
            encoded, tuple(representatives), dispositions, **options)
    try:
        return construct(result, proof_json, representatives)
    except _RecordExhausted as error:
        refused = _byte_refusal(result, error)
        return construct(refused, _proof_json(refused), ())


class _ValidationWork(csp._Work):
    """Independent certificate-validation pass, one meter across all choices.

    Hard record-shape/digest limits apply even to a budget_refused certificate.
    Interpretation cue/relative-axis/connectivity work uses its stored CSP
    limits and must fail closed cumulatively, never reset per cue or choice.
    These validation charges are separate from persisted Phase A search counts.
    """
    def charge(self, name, amount):
        try:
            return super().charge(name, amount)
        except csp._Exhausted as error:
            raise ValueError(f"endpoint validation budget refused: {name}") from error

    def bound(self, name, amount):
        try:
            return super().bound(name, amount)
        except csp._Exhausted as error:
            raise ValueError(f"endpoint validation budget refused: {name}") from error


def validate_candidate_certificate(cert, readings, signatures, candidate_matches):
    """Recompute the complete bounded proof; source authenticity requires replay."""
    partial = type(cert) is PartialEndpointSelectionCertificate
    if type(cert) is not EndpointSelectionCertificate and not partial:
        raise ValueError("foreign endpoint selection certificate")
    type(cert)(**legacy._fields(cert))
    if (cert.reading_bindings != reading_bindings(readings, signatures)
            or cert.candidate_count != len(candidate_matches)
            or cert.candidate_matches_sha256 != candidate_digest(candidate_matches)):
        raise ValueError("endpoint candidate/reading/signature bindings differ")
    canonical, _, _ = plane_graph._canonical_reading_index(readings)
    for row in candidate_matches:
        plane_graph._validate_indexed_match(row, canonical)
    proof = cert.proof_record()
    work = _ValidationWork(cert.budget)
    for choice in proof["choices"]:
        q = dict(choice["quadrants"])
        if any((q[r] - q[l]) % 4 != relative for l, r, relative in cert.quadrant_constraints):
            raise ValueError("selected assignment contradicts quadrant constraints")
        selected = {(f, i): a for f, i, a in choice["alternatives"]}
        supports = {}
        for row_index, row in enumerate(candidate_matches):
            work.charge("relation_visits", 1)
            if (q[row.right] - q[row.left]) % 4 != row.quadrant_of_right:
                continue
            left, right = (row.left, row.left_plane), (row.right, row.right_plane)
            u, v = sorted((left, right))
            supports.setdefault((u, v), {})
            work.charge("relation_visits", len(canonical[left].alternatives) + len(canonical[right].alternatives))
            if (canonical[left].alternatives[selected[left]] not in canonical[left].matching_alternatives(row.left_b)
                    or canonical[right].alternatives[selected[right]] not in canonical[right].matching_alternatives(row.right_b)):
                continue
            cue = {name: value for name, value in csp._row_payload(row).items()
                   if name not in ("left", "right", "left_axis", "right_axis", "left_plane",
                                   "right_plane", "left_b", "right_b", "quadrant_of_right")}
            cue_digest = hashlib.sha256(work.encode(cue)).hexdigest()
            supports[(u, v)].setdefault(cue_digest, []).append(row_index)
        expected = []
        for (u, v), groups in sorted(supports.items()):
            rows = sorted(i for group in groups.values() for i in group)
            if not rows:
                raise ValueError("selected assignment omits a mandatory factor")
            expected.append({"endpoints": [list(u), list(v)], "alternatives": [selected[u], selected[v]],
                             "source_rows": rows, "distinguishable_support_groups":
                             [group for _, group in sorted(groups.items())]})
        if choice["factors"] != expected:
            raise ValueError("mandatory factor/evidence source-row support differs")
        factors = [(u, v, {}) for u, v in sorted(supports)]
        eligibility_reason = (partial_axes.classify(work, factors, canonical, tuple(sorted(cert.frame_ids)), q,
            _partial_anchor(cert.reading_bindings, cert.frame_ids))[0] if partial else
            csp._eligibility_reason(work, factors, canonical, tuple(sorted(cert.frame_ids)), q))
        if eligibility_reason:
            raise ValueError("selected assignment is structurally ineligible")
    if cert.status == "unique":
        for rep in cert.representatives:
            original = candidate_matches[rep.source_row]
            if dataclasses.replace(original, left_b=rep.match.left_b, right_b=rep.match.right_b) != rep.match:
                raise ValueError("representative match is not its exact source-row evidence")
    # Exact recomputation binds all outcomes, counts, exclusions, cycle witnesses,
    # selected indices and physical relations. A caller's resealed hash is no proof.
    replay = csp.solve_endpoint_hypotheses(readings, candidate_matches,
        session_frames=cert.frame_ids, quadrant_constraints=cert.quadrant_constraints, budget=cert.budget,
        **(dict(_observability_mode="partial") if partial else {}))
    expected = _finish_selection(replay, canonical, candidate_matches, cert.reading_bindings,
        cert.candidate_matches_sha256, cert.candidate_dispositions, frame_ids=cert.frame_ids,
        quadrant_constraints=cert.quadrant_constraints, budget=cert.budget,
        session_sha256=cert.session_sha256, lattice_sha256=cert.lattice_sha256,
        signatures_sha256=cert.signatures_sha256, connection_sha256=cert.connection_sha256,
        _partial_mode=partial)
    if expected.proof_json != cert.proof_json or expected.result_sha256 != cert.result_sha256:
        raise ValueError("endpoint physical proof does not recompute")
