"""DSCALE-29: one-shot feasibility in a pinned, captured candidate domain.

No reconstruction modules are imported. Axis signs reproduce plane_graph's
documented Ry(-90*q) table. Translation feasibility is a difference-constraints
problem, not a least-squares rejection test. Fractions make its decisions exact
for the captured binary floating-point distances and the decimal 0.15 bound.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

INPUT_SHA256 = "5b34aac34bc7c889c48248ccbf52460382519c9d176ae13a77ae6c33076ab1a2"
CAPTURE_SHA256 = "744cc05bbecce192ba9d4bf33c5306b670eebf068c82409218319ae7b11a352d"
BINDINGS_SHA256 = "2c9855bc49e82e3544045b0457a85d6fa74c544ed442d5a8ea291c7629a60e85"
CONTEXT_SHA256 = "f569744c97308f3f878612a2763187f7e1c071a962e3e9025b047760db962475"
ASSIGNMENT_SHA256 = "4450b351848e894ac0afac7480b74c36d896f38b32962187a114fc740457ecba"
MATCH_SHA256 = "1e392694dc12b3000280742720cbcd14c5dbd8ad7e61ab02dc36e51ad844ad5a"
AXIS_TABLE = (
    ((0, 1), (1, 1), (2, 1)),
    ((2, 1), (1, 1), (0, -1)),
    ((0, -1), (1, 1), (2, -1)),
    ((2, -1), (1, 1), (0, 1)),
)
TOLERANCE = Fraction("0.15")
DEFAULT_INPUT = "exports/diagnostics/plane-residual-observer-2026-10-07/observation.json"
DEFAULT_OUTPUT = "exports/diagnostics/plane-residual-observer-2026-10-07/three-axis-feasibility.json"
BOUNDARY = ("A witness proves existence only within these captured groups and fixed "
            "assignment. It does not prove correct camera or period, uniqueness, global "
            "candidate completeness, solver ambiguity policy, DSCALE-12 passage, geometry, "
            "image accuracy, or production acceptance. C1/C2 remain closed.")


@dataclass(frozen=True)
class Limits:
    nodes: int = 262144
    assignments: int = 8192
    seconds: float = 60.0
    rss_bytes: int = 2147483648
    output_bytes: int = 2097152


@dataclass(frozen=True)
class Candidate:
    source_index: int
    group: int
    ordinal: int
    left: tuple[str, int]
    right: tuple[str, int]
    left_b: float
    right_b: float
    axis: int
    delta: Fraction
    disposition: str
    score: float


class Incomplete(Exception):
    pass


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def integer(value, low, high):
    return type(value) is int and low <= value <= high


def rss_usage():
    """Return peak RSS with standard-library OS APIs; fail closed if unavailable."""
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak", "working", "peak_paged", "paged", "peak_nonpaged",
                    "nonpaged", "pagefile", "peak_pagefile")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise Incomplete("rss_measurement_unavailable")
        return counters.peak
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


class Budget:
    def __init__(self, limits=Limits(), clock=time.monotonic, memory=rss_usage):
        self.limits, self.clock, self.memory = limits, clock, memory
        self.started = clock()
        self.peak_rss = 0
        self.counters = dict(prefix_expansion_nodes=0, b_conflict_rejections=0,
                             complete_b_consistent_assignments=0,
                             connectivity_rejections=0, residual_rejections=0)

    def check(self):
        if self.clock() - self.started >= self.limits.seconds:
            raise Incomplete("wall_deadline")
        self.peak_rss = max(self.peak_rss, self.memory())
        if self.peak_rss > self.limits.rss_bytes:
            raise Incomplete("rss_cap")

    def node(self):
        self.check()
        if self.counters["prefix_expansion_nodes"] >= self.limits.nodes:
            raise Incomplete("prefix_expansion_node_cap")
        self.counters["prefix_expansion_nodes"] += 1

    def complete(self):
        self.check()
        if self.counters["complete_b_consistent_assignments"] >= self.limits.assignments:
            raise Incomplete("complete_assignment_cap")
        self.counters["complete_b_consistent_assignments"] += 1


def validate(payload, budget):
    """Validate every row and binding. Never salvage a partial malformed domain."""
    capture = payload["same_axis_candidates"]
    failure = payload["solver_failure_observation"]
    bindings = failure["solver_input_bindings"]
    raw = failure["raw"]
    context = raw["solve_context"]
    for record, schema, digest, basis in (
        (capture, "mc-photogrammetry-same-axis-candidate-capture-v2", CAPTURE_SHA256,
         "derived_from_active_pre_same_axis_candidates"),
        (bindings, "mc-photogrammetry-solver-input-bindings-v1", BINDINGS_SHA256,
         "derived_from_post_same_axis_solver_inputs"),
        (context, "mc-photogrammetry-solver-solve-context-v1", CONTEXT_SHA256,
         "derived_from_effective_solve_component_call_arguments"),
    ):
        require(record["schema"] == schema and record["status"] == "complete"
                and record["record_sha256"] == digest and record["basis"] == basis,
                "invalid_record_binding")
    frames = capture["frames"]
    yaw = capture["assignments"]
    require(frames == ["image-0000", "image-0001", "image-0002", "image-0003"]
            and frames == bindings["frames"] == context["frames"] == raw["frames"],
            "invalid_frame_binding")
    require(yaw == [0, 3, 3, 1] and yaw == raw["quadrants"]
            and all(integer(q, 0, 3) for q in yaw), "invalid_fixed_assignment")
    require(capture["axes"] == [0, 1, 2] and context["integer_tolerance"] == 0.15,
            "invalid_axes_or_tolerance")
    for record in (capture, bindings):
        require(record["failed_assignment_sha256"] == ASSIGNMENT_SHA256
                and record["input_match_identity_sha256"] == MATCH_SHA256,
                "invalid_cross_record_identity")
    require(capture["readings"] == bindings["readings"], "reading_binding_mismatch")
    readings = {}
    keys = set()
    require(len(bindings["readings"]) == 18, "missing_readings")
    for expected_ref, reading in enumerate(bindings["readings"]):
        budget.check()
        require(type(reading) is list and len(reading) == 6, "invalid_reading_layout")
        ref, frame, plane, axis, side, alternatives = reading
        require(type(ref) is int and ref == expected_ref and frame in frames
                and integer(plane, 0, 1000000) and integer(axis, 0, 2)
                and type(side) is int and side in (-1, 1), "invalid_reading")
        require((frame, plane) not in keys and type(alternatives) is list
                and len(alternatives) > 0, "duplicate_or_empty_reading")
        seen_b = set()
        for alternative in alternatives:
            require(type(alternative) is list and len(alternative) == 3
                    and all(finite(v) for v in alternative), "invalid_alternative")
            b, distance, support = alternative
            require(b > 0 and distance > 0 and 0 <= support <= 1 and b not in seen_b,
                    "invalid_or_ambiguous_alternative")
            seen_b.add(b)
        readings[ref] = reading
        keys.add((frame, plane))
    quadrant = dict(zip(frames, yaw))

    def endpoint(ref, alt):
        require(integer(ref, 0, len(readings)-1), "invalid_reading_ref")
        reading = readings[ref]
        require(integer(alt, 0, len(reading[5])-1), "invalid_alternative_ref")
        frame, plane, axis, side = reading[1:5]
        b, distance, support = reading[5][alt]
        world, sign = AXIS_TABLE[quadrant[frame]][axis]
        return (frame, plane), b, Fraction(distance)*side*sign, world, [
            ref, alt, b, distance, support, side, sign, axis]

    identities, edge_identities, active = capture["input_identities"], [], {}
    require(capture["input_match_count"] == bindings["input_match_count"] == 70
            and len(identities) == len(bindings["edges"]) == 70,
            "invalid_input_match_count")
    for index, edge in enumerate(bindings["edges"]):
        budget.check()
        require(type(edge) is list and len(edge) == 5, "invalid_edge_layout")
        match, left, right, relative, enabled = edge
        require(type(match) is int and match == index and type(enabled) is bool
                and integer(relative, 0, 3), "invalid_edge")
        require(type(left) is list and type(right) is list
                and len(left) == len(right) == 8, "invalid_edge_endpoint")
        le = endpoint(left[0], left[1]); re = endpoint(right[0], right[1])
        require(left == le[4] and right == re[4], "invalid_endpoint_binding")
        identity = [*le[0], left[7], *re[0], right[7], relative, le[1], re[1]]
        edge_identities.append(identity)
        if enabled:
            require((quadrant[re[0][0]]-quadrant[le[0][0]]) % 4 == relative
                    and le[3] == re[3], "invalid_active_edge_assignment")
            active[index] = identity
    require(identities == edge_identities and len(active) == bindings["active_edge_count"] == 16,
            "invalid_match_identity_binding")
    groups = capture["groups"]
    require(capture["group_count"] == len(groups) == 16, "missing_group_obligation")
    group_keys, group_matches = set(), set()
    for index, group in enumerate(groups):
        require(type(group) is list and len(group) == 7, "invalid_group_layout")
        gid, left, right, axis, relative, matches, declared = group
        require(type(gid) is int and gid == index and left in frames and right in frames
                and left != right and integer(axis, 0, 2) and integer(relative, 0, 3),
                "invalid_group_identity")
        require((quadrant[right]-quadrant[left]) % 4 == relative,
                "group_assignment_mismatch")
        require(type(matches) is list and len(matches) == 1 and type(matches[0]) is int
                and matches[0] in active and declared == [active[matches[0]]],
                "invalid_group_survivor_binding")
        identity = declared[0]
        require(identity[0] == left and identity[3] == right and identity[2] == axis
                and identity[6] == relative, "group_identity_mismatch")
        key = (left, right, axis, relative)
        require(key not in group_keys and matches[0] not in group_matches,
                "duplicate_group")
        group_keys.add(key); group_matches.add(matches[0])
    require(group_matches == set(active), "unbound_active_group")
    rows = capture["candidates"]
    require(capture["row_count"] == len(rows) == 437, "missing_candidate_rows")
    candidates = [[] for _ in groups]
    kept = [[] for _ in groups]
    for source_index, row in enumerate(rows):
        budget.check()
        require(type(row) is list and len(row) == 19, "invalid_candidate_layout")
        # Producer order (image_derived_graph._collect_same_axis_candidates):
        # fields 3..6 are left_reading_ref, left_alternative_index,
        # right_reading_ref, right_alternative_index.
        gid, match, ordinal, lr, la, rr, ra, relative = row[:8]
        require(integer(gid, 0, 15) and type(ordinal) is int
                and ordinal == len(candidates[gid]), "invalid_source_row_identity")
        group = groups[gid]
        require(integer(lr, 0, len(readings)-1) and integer(rr, 0, len(readings)-1),
                "invalid_reading_ref")
        le, re = endpoint(lr, la), endpoint(rr, ra)
        require(le[0][0] == group[1] and re[0][0] == group[2]
                and readings[lr][3] == group[3] and relative == group[4]
                and type(relative) is int and le[3] == re[3], "candidate_group_mismatch")
        require(all(finite(row[i]) for i in (8, 9, 11, 12, 16))
                and type(row[10]) is bool and integer(row[13], 0, 1000000000)
                and type(row[14]) is int and type(row[15]) is int,
                "invalid_candidate_measurement")
        disposition, reason = row[17:19]
        require(type(match) is int and ((disposition == "dropped" and match == -1
                 and reason == "same_axis_mismatch") or
                (disposition == "kept" and match == group[5][0] and reason == "")),
                "invalid_candidate_disposition")
        if disposition == "kept":
            identity = [*le[0], readings[lr][3], *re[0], readings[rr][3], relative, le[1], re[1]]
            require(identity == active[match], "kept_candidate_binding_mismatch")
            kept[gid].append(match)
        candidates[gid].append(Candidate(source_index, gid, ordinal, le[0], re[0],
                                         le[1], re[1], le[3], le[2]-re[2], disposition, row[8]))
    require(all(candidates) and all(len(items) == 1 for items in kept)
            and sum(c.disposition == "dropped" for items in candidates for c in items) == 421,
            "missing_or_duplicate_group_candidates")
    return frames, yaw, groups, candidates


def connected(frames, edges):
    reached = {frames[0]}
    for _ in frames:
        for edge in edges:
            if edge.left[0] in reached or edge.right[0] in reached:
                reached.update((edge.left[0], edge.right[0]))
    return reached == set(frames)


def potentials(frames, edges, tolerance=TOLERANCE):
    """Exact Bellman-Ford feasibility for |t_right-t_left-delta| <= tolerance."""
    values = dict.fromkeys(frames, Fraction(0))
    constraints = []
    for edge in edges:
        left, right = edge.left[0], edge.right[0]
        constraints.extend(((left, right, edge.delta+tolerance),
                            (right, left, -edge.delta+tolerance)))
    for _ in frames:
        changed = False
        for left, right, upper in constraints:
            if values[right] > values[left] + upper:
                values[right] = values[left] + upper
                changed = True
        if not changed:
            anchor = values[frames[0]]
            return {frame: value-anchor for frame, value in values.items()}
    return None  # A negative cycle proves infeasibility of these interval equations.


def exact(value):
    return f"{value.numerator}/{value.denominator}"


def axis_certificate(frames, edges, values):
    # Tree potentials supply fundamental cycle sums independently of feasible values.
    tree_values, tree_paths = {frames[0]: Fraction(0)}, {frames[0]: []}
    tree_sources = set()
    while len(tree_values) < len(frames):
        for edge in edges:
            left, right = edge.left[0], edge.right[0]
            if (left in tree_values) == (right in tree_values):
                continue
            known, new, sign = (left, right, 1) if left in tree_values else (right, left, -1)
            tree_values[new] = tree_values[known] + sign*edge.delta
            tree_paths[new] = tree_paths[known] + [[edge.source_index, sign]]
            tree_sources.add(edge.source_index)
    cycles, equation_rows, residuals = [], [], []
    for edge in edges:
        left, right = edge.left[0], edge.right[0]
        residual = values[right]-values[left]-edge.delta
        residuals.append(abs(residual))
        equation_rows.append(dict(source_index=edge.source_index, group=edge.group,
                                  left=left, right=right, delta=float(edge.delta),
                                  delta_exact=exact(edge.delta), residual=float(residual),
                                  residual_exact=exact(residual)))
        if edge.source_index not in tree_sources:
            cycle_sum = tree_values[left]+edge.delta-tree_values[right]
            cycles.append(dict(terms=tree_paths[left]+[[edge.source_index, 1]]+
                               [[idx, -sign] for idx, sign in reversed(tree_paths[right])],
                               sum=float(cycle_sum), sum_exact=exact(cycle_sum)))
    maximum = max(residuals, default=Fraction(0))
    require(maximum <= TOLERANCE, "internal_certificate_residual_failure")
    return dict(connected_to_anchor=True, edges=equation_rows, cycles=cycles,
                anchored_translations={f: float(v) for f, v in values.items()},
                anchored_translations_exact={f: exact(v) for f, v in values.items()},
                maximum_absolute_edge_residual=float(maximum),
                maximum_absolute_edge_residual_exact=exact(maximum))


def search(frames, groups, budget):
    # Every candidate is visited in group-size then captured source identity order.
    order = sorted(range(len(groups)), key=lambda gid: (len(groups[gid]), gid))
    endpoints, selected = {}, []

    def visit(depth):
        budget.check()
        if depth == len(order):
            budget.complete()
            axes = [[row for row in selected if row.axis == axis] for axis in range(3)]
            if not all(connected(frames, edges) for edges in axes):
                budget.counters["connectivity_rejections"] += 1
                return None
            values = [potentials(frames, edges) for edges in axes]
            if any(value is None for value in values):
                budget.counters["residual_rejections"] += 1
                return None
            return dict(selected_source_rows=[dict(source_index=row.source_index,
                        group=row.group, ordinal=row.ordinal, disposition=row.disposition,
                        score=row.score, left=list(row.left), right=list(row.right),
                        left_b=row.left_b, right_b=row.right_b, world_axis=row.axis)
                        for row in sorted(selected, key=lambda row: row.group)],
                        endpoint_b_map=[[*key, b] for key, b in sorted(endpoints.items())],
                        axes=[dict(axis=axis, **axis_certificate(frames, edges, value))
                              for axis, (edges, value) in enumerate(zip(axes, values))])
        for row in groups[order[depth]]:
            budget.node()
            bindings = ((row.left, row.left_b), (row.right, row.right_b))
            if any(key in endpoints and endpoints[key] != b for key, b in bindings):
                budget.counters["b_conflict_rejections"] += 1
                continue
            added = [key for key, b in bindings if key not in endpoints]
            for key, b in bindings:
                endpoints[key] = b
            selected.append(row)
            result = visit(depth+1)
            selected.pop()
            for key in added:
                del endpoints[key]
            if result is not None:
                return result
        return None

    try:
        witness = visit(0)
        return ("feasible_within_captured_domain" if witness else
                "infeasible_within_captured_domain"), None, witness, order
    except Incomplete as error:
        return "incomplete", str(error), None, order


def run(input_path, limits=Limits(), clock=time.monotonic, memory=rss_usage):
    budget = Budget(limits, clock, memory)
    result = dict(schema="mc-photogrammetry-captured-three-axis-feasibility-v1",
                  status="incomplete", reason=None, input_path=str(input_path),
                  expected_input_sha256=INPUT_SHA256, input_sha256=None,
                  tolerance_blocks=0.15, tolerance_exact="3/20", anchor="image-0000",
                  limits=vars(limits), counters=budget.counters, witness=None,
                  interpretation=BOUNDARY,
                  counter_semantics={
                      "prefix_expansion_nodes": "Each candidate row considered at a prefix, including b conflicts; root is not counted.",
                      "complete_b_consistent_assignments": "Every complete one-row-per-group exact-b-consistent choice examined."},
                  equation_convention="t_right - t_left = sign_left*side_left*distance_left - sign_right*side_right*distance_right",
                  arithmetic="exact rationals of captured binary float distances; decimal tolerance 3/20")
    try:
        budget.check()
        raw = Path(input_path).read_bytes()
        result["input_sha256"] = hashlib.sha256(raw).hexdigest()
        require(result["input_sha256"] == INPUT_SHA256, "input_sha256_mismatch")
        budget.check()
        payload = json.loads(raw, parse_constant=lambda v: (_ for _ in ()).throw(ValueError("nonfinite_json")))
        frames, yaw, groups, candidates = validate(payload, budget)
        result.update(input_validation="complete", frames=frames, captured_assignment=yaw,
                      captured_groups=groups, candidate_rows=437, kept_rows=16, dropped_rows=421,
                      record_bindings=dict(capture=CAPTURE_SHA256, solver_input_bindings=BINDINGS_SHA256,
                                           solve_context=CONTEXT_SHA256,
                                           failed_assignment=ASSIGNMENT_SHA256,
                                           input_match_identity=MATCH_SHA256),
                      group_candidate_counts=[len(items) for items in candidates])
        status, reason, witness, order = search(frames, candidates, budget)
        result.update(status=status, reason=reason, witness=witness, search_group_order=order,
                      exhaustive=(status == "infeasible_within_captured_domain"))
    except Incomplete as error:
        result["reason"] = str(error)
    except (ValueError, KeyError, TypeError, IndexError, OSError) as error:
        result["reason"] = "input_validation: " + str(error)
    result.update(elapsed_seconds=clock()-budget.started, peak_rss_bytes=budget.peak_rss)
    return result


def encode_result(result, limit):
    data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n").encode("utf-8")
    if len(data) > limit:
        result.update(status="incomplete", reason="output_size_cap", witness=None, exhaustive=False)
        data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n").encode("utf-8")
    if len(data) > limit:
        raise Incomplete("output_size_cap_cannot_write_receipt")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.input)
    data = encode_result(result, Limits().output_bytes)
    Path(args.output).write_bytes(data)
    print(json.dumps({key: result[key] for key in
                      ("status", "reason", "input_sha256", "counters", "elapsed_seconds", "peak_rss_bytes")}))
    print(json.dumps(dict(output_bytes=len(data), output_sha256=hashlib.sha256(data).hexdigest())))
    return 0 if result["status"] != "incomplete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
