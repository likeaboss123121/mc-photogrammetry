"""Synthetic DSCALE-29 checks; run directly to avoid project conftest imports."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from fractions import Fraction

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/diagnostics/captured_three_axis_feasibility.py"
SPEC = importlib.util.spec_from_file_location("captured_three_axis_feasibility", MODULE_PATH)
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def candidate(index, group, axis, delta=0, b=1, left="image-0000", right="other",
              left_plane=None, right_plane=None, score=1, disposition="kept"):
    return checker.Candidate(index, group, index, (left, axis if left_plane is None else left_plane),
                             (right, axis if right_plane is None else right_plane),
                             b, b, axis, Fraction(delta), disposition, score)


def budget(**kwargs):
    return checker.Budget(checker.Limits(**kwargs), clock=lambda: 0, memory=lambda: 10)


# Synthetic capture mirroring the producer row layout in
# image_derived_graph._collect_same_axis_candidates: row fields 3..6 are
# left_reading_ref, left_local_alternative_index, right_reading_ref,
# right_local_alternative_index. Candidate rows deliberately encode references
# and local alternative indices, never plane numbers; plane ids are 100+ref so
# the obsolete plane-key remap cannot resolve them.
SYNTHETIC_FRAMES = ["image-0000", "image-0001", "image-0002", "image-0003"]
SYNTHETIC_QUADRANTS = [0, 3, 3, 1]
SYNTHETIC_AXIS_BY_REF = {5: 0, 8: 2, 9: 0, 11: 2}
SYNTHETIC_REFS_BY_FRAME = {
    "image-0000": (0, 1, 2, 3, 4),
    "image-0001": (5, 6, 7, 8),
    "image-0002": (9, 10, 11, 12),
    "image-0003": (13, 14, 15, 16, 17),
}


def synthetic_readings():
    readings = []
    for frame, refs in SYNTHETIC_REFS_BY_FRAME.items():
        for ref in refs:
            readings.append([ref, frame, 100 + ref, SYNTHETIC_AXIS_BY_REF.get(ref, 1), 1,
                             [[1.0 + ref, 0.5 + ref, 0.5],
                              [2.0 + ref, 0.25 + ref, 0.25]]])
    readings.sort(key=lambda reading: reading[0])
    return readings


def synthetic_endpoint(readings, ref, alt):
    reading = readings[ref]
    frame, _plane, axis, side = reading[1:5]
    b, distance, support = reading[5][alt]
    world, sign = checker.AXIS_TABLE[
        SYNTHETIC_QUADRANTS[SYNTHETIC_FRAMES.index(frame)]][axis]
    return ((frame, reading[2]), b, Fraction(distance)*side*sign, world,
            [ref, alt, b, distance, support, side, sign, axis])


def synthetic_identity(readings, ref, alt, right_ref, right_alt, relative):
    left = synthetic_endpoint(readings, ref, alt)
    right = synthetic_endpoint(readings, right_ref, right_alt)
    return [left[0][0], left[0][1], left[4][7], right[0][0], right[0][1],
            right[4][7], relative, left[1], right[1]]


def synthetic_group_plan():
    first_axis_one_ref = {0: 1, 1: 6, 2: 10, 3: 13}
    pairs = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3),
             (2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2)]
    plan = [(left, right, 1, first_axis_one_ref[left], first_axis_one_ref[right], 0, 1)
            for left, right in pairs]
    plan.extend([(1, 2, 0, 5, 9, 0, 1), (2, 1, 0, 9, 5, 0, 1),
                 (1, 2, 2, 8, 11, 0, 1), (2, 1, 2, 11, 8, 0, 1)])
    return plan


def synthetic_row(gid, match, ordinal, lref, lalt, rref, ralt, relative,
                  disposition, reason):
    return [gid, match, ordinal, lref, lalt, rref, ralt, relative,
            1.0 + ordinal, 0.5, False, 2.0, 0.25, 3, 4, 5, 0.125,
            disposition, reason]


def synthetic_payload():
    readings = synthetic_readings()
    plan = synthetic_group_plan()
    relative_of = lambda left, right: (
        SYNTHETIC_QUADRANTS[right] - SYNTHETIC_QUADRANTS[left]) % 4
    edges, identities = [], []
    for index, (left, right, _axis, lref, rref, lalt, ralt) in enumerate(plan):
        relative = relative_of(left, right)
        edges.append([index, synthetic_endpoint(readings, lref, lalt)[4],
                      synthetic_endpoint(readings, rref, ralt)[4], relative, True])
        identities.append(synthetic_identity(readings, lref, lalt, rref, ralt, relative))
    for index in range(len(plan), 70):
        lref, rref, relative = (index*3) % 18, (index*3 + 7) % 18, index % 4
        edges.append([index, synthetic_endpoint(readings, lref, 0)[4],
                      synthetic_endpoint(readings, rref, 1)[4], relative, False])
        identities.append(synthetic_identity(readings, lref, 0, rref, 1, relative))
    groups, pools, rows = [], {}, []
    for reading in readings:
        pools.setdefault((reading[1], reading[3]), []).append(reading[0])
    for gid, (left, right, axis, lref, rref, lalt, ralt) in enumerate(plan):
        relative = relative_of(left, right)
        groups.append([gid, SYNTHETIC_FRAMES[left], SYNTHETIC_FRAMES[right], axis,
                       relative, [gid],
                       [synthetic_identity(readings, lref, lalt, rref, ralt, relative)]])
        rows.append(synthetic_row(gid, gid, 0, lref, lalt, rref, ralt, relative,
                                  "kept", ""))
        left_pool = pools[(SYNTHETIC_FRAMES[left], axis)]
        right_pool = pools[(SYNTHETIC_FRAMES[right], axis)]
        for ordinal in range(1, 32 if gid == 0 else 27):
            rows.append(synthetic_row(
                gid, -1, ordinal, left_pool[ordinal % len(left_pool)], ordinal % 2,
                right_pool[(ordinal + 1) % len(right_pool)], (ordinal + 1) % 2,
                relative, "dropped", "same_axis_mismatch"))
    capture = dict(schema="mc-photogrammetry-same-axis-candidate-capture-v2",
                   status="complete", record_sha256=checker.CAPTURE_SHA256,
                   basis="derived_from_active_pre_same_axis_candidates",
                   failed_assignment_sha256=checker.ASSIGNMENT_SHA256,
                   input_match_identity_sha256=checker.MATCH_SHA256,
                   frames=list(SYNTHETIC_FRAMES), assignments=list(SYNTHETIC_QUADRANTS),
                   axes=[0, 1, 2], readings=readings, groups=groups, candidates=rows,
                   input_identities=identities, input_match_count=70,
                   group_count=16, row_count=437)
    bindings = dict(schema="mc-photogrammetry-solver-input-bindings-v1",
                    status="complete", record_sha256=checker.BINDINGS_SHA256,
                    basis="derived_from_post_same_axis_solver_inputs",
                    failed_assignment_sha256=checker.ASSIGNMENT_SHA256,
                    input_match_identity_sha256=checker.MATCH_SHA256,
                    frames=list(SYNTHETIC_FRAMES), readings=readings, edges=edges,
                    input_match_count=70, active_edge_count=16)
    context = dict(schema="mc-photogrammetry-solver-solve-context-v1",
                   status="complete", record_sha256=checker.CONTEXT_SHA256,
                   basis="derived_from_effective_solve_component_call_arguments",
                   frames=list(SYNTHETIC_FRAMES), integer_tolerance=0.15)
    return dict(same_axis_candidates=capture,
                solver_failure_observation=dict(
                    solver_input_bindings=bindings,
                    raw=dict(solve_context=context, frames=list(SYNTHETIC_FRAMES),
                             quadrants=list(SYNTHETIC_QUADRANTS))))


class FeasibilityTests(unittest.TestCase):
    def test_all_axes_and_anchored_equations(self):
        groups = [[candidate(axis, axis, axis, axis+1)] for axis in range(3)]
        status, reason, witness, order = checker.search(["image-0000", "other"], groups, budget())
        self.assertEqual(status, "feasible_within_captured_domain")
        self.assertIsNone(reason)
        self.assertEqual(order, [0, 1, 2])
        for axis in witness["axes"]:
            values = {frame: Fraction(value) for frame, value in axis["anchored_translations_exact"].items()}
            self.assertEqual(values["image-0000"], 0)
            for edge in axis["edges"]:
                residual = values[edge["right"]]-values[edge["left"]]-Fraction(edge["delta_exact"])
                self.assertLessEqual(abs(residual), Fraction("0.15"))

    def test_exact_shared_endpoint_b_rejects_near_equal(self):
        groups = [[candidate(0, 0, 0)], [candidate(1, 1, 0, b=1+1e-12)],
                  [candidate(2, 2, 1)], [candidate(3, 3, 2)]]
        work = budget()
        status, _, witness, _ = checker.search(["image-0000", "other"], groups, work)
        self.assertEqual(status, "infeasible_within_captured_domain")
        self.assertIsNone(witness)
        self.assertEqual(work.counters["b_conflict_rejections"], 1)
        self.assertEqual(work.counters["complete_b_consistent_assignments"], 0)

    def test_connectivity_required_independently_on_each_axis(self):
        groups = [[candidate(0, 0, 0)], [candidate(1, 1, 1)]]
        work = budget()
        self.assertEqual(checker.search(["image-0000", "other"], groups, work)[0],
                         "infeasible_within_captured_domain")
        self.assertEqual(work.counters["connectivity_rejections"], 1)

    def test_missing_frame_is_disconnected_even_with_all_axes(self):
        groups = [[candidate(i, i, i)] for i in range(3)]
        self.assertEqual(checker.search(["image-0000", "other", "isolated"], groups, budget())[0],
                         "infeasible_within_captured_domain")

    def test_residual_is_unwrapped(self):
        groups = [[candidate(0, 0, 0)], [candidate(1, 1, 0, 1)],
                  [candidate(2, 2, 1)], [candidate(3, 3, 2)]]
        work = budget()
        self.assertEqual(checker.search(["image-0000", "other"], groups, work)[0],
                         "infeasible_within_captured_domain")
        self.assertEqual(work.counters["residual_rejections"], 1)

    def test_lower_score_dropped_row_retained(self):
        groups = [[candidate(0, 0, 0, 1, score=100),
                   candidate(1, 0, 0, 0, score=-100, disposition="dropped")],
                  [candidate(2, 1, 0)], [candidate(3, 2, 1)], [candidate(4, 3, 2)]]
        work = budget()
        status, _, witness, _ = checker.search(["image-0000", "other"], groups, work)
        self.assertEqual(status, "feasible_within_captured_domain")
        self.assertEqual(witness["selected_source_rows"][0]["source_index"], 1)
        self.assertEqual(work.counters["residual_rejections"], 1)
        self.assertEqual(work.counters["complete_b_consistent_assignments"], 2)

    def test_interval_feasibility_can_succeed_when_least_squares_max_fails(self):
        # LS would average [0,0,.3] to .1: maximum residual .2. The minimax
        # translation .15 satisfies every edge, including the exact boundary.
        edges = [candidate(0, 0, 0), candidate(1, 1, 0),
                 candidate(2, 2, 0, Fraction("0.3"))]
        values = checker.potentials(["image-0000", "other"], edges)
        self.assertIsNotNone(values)
        self.assertEqual(values["other"], Fraction("0.15"))
        self.assertIsNone(checker.potentials(["image-0000", "other"],
                          edges+[candidate(3, 3, 0, Fraction("0.300000000001"))]))

    def test_cycle_sums_and_signed_edges(self):
        frames = ["image-0000", "b", "c"]
        edges = [candidate(0, 0, 0, 2, right="b"),
                 candidate(1, 1, 0, -3, left="b", right="c"),
                 candidate(2, 2, 0, -1, right="c")]
        values = checker.potentials(frames, edges)
        certificate = checker.axis_certificate(frames, edges, values)
        self.assertEqual(len(certificate["cycles"]), 1)
        self.assertEqual(Fraction(certificate["cycles"][0]["sum_exact"]), 0)
        deltas = {edge.source_index: edge.delta for edge in edges}
        for cycle in certificate["cycles"]:
            self.assertEqual(sum(sign*deltas[index] for index, sign in cycle["terms"]),
                             Fraction(cycle["sum_exact"]))

    def test_node_cap_is_typed_incomplete_and_exact(self):
        work = budget(nodes=1)
        groups = [[candidate(i, i, i)] for i in range(3)]
        result = checker.search(["image-0000", "other"], groups, work)
        self.assertEqual(result[:3], ("incomplete", "prefix_expansion_node_cap", None))
        self.assertEqual(work.counters["prefix_expansion_nodes"], 1)

    def test_assignment_cap_is_typed_incomplete_and_exact(self):
        groups = [[candidate(0, 0, 0, 1), candidate(1, 0, 0, 0)],
                  [candidate(2, 1, 0)], [candidate(3, 2, 1)], [candidate(4, 3, 2)]]
        work = budget(assignments=1)
        result = checker.search(["image-0000", "other"], groups, work)
        self.assertEqual(result[:3], ("incomplete", "complete_assignment_cap", None))
        self.assertEqual(work.counters["complete_b_consistent_assignments"], 1)

    def test_deadline_is_typed_incomplete(self):
        work = budget(seconds=0)
        result = checker.search(["image-0000", "other"], [[candidate(0, 0, 0)]], work)
        self.assertEqual(result[:3], ("incomplete", "wall_deadline", None))
        self.assertEqual(work.counters["prefix_expansion_nodes"], 0)

    def test_rss_cap_is_typed_incomplete(self):
        work = budget(rss_bytes=9)
        result = checker.search(["image-0000", "other"], [[candidate(0, 0, 0)]], work)
        self.assertEqual(result[:3], ("incomplete", "rss_cap", None))
        self.assertEqual(work.peak_rss, 10)

    def test_cli_loader_checks_hash_before_parsing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"invalid.json"
            path.write_bytes(b"not JSON")
            result = checker.run(path, clock=lambda: 0, memory=lambda: 10)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["reason"], "input_validation: input_sha256_mismatch")
        self.assertEqual(result["counters"]["prefix_expansion_nodes"], 0)

    def test_output_size_cap_has_incomplete_receipt(self):
        result = dict(status="feasible_within_captured_domain", reason=None,
                      witness={"large": "x"*1000}, counters={"nodes": 7})
        data = checker.encode_result(result, 300)
        receipt = json.loads(data)
        self.assertLessEqual(len(data), 300)
        self.assertEqual(receipt["status"], "incomplete")
        self.assertEqual(receipt["reason"], "output_size_cap")
        self.assertEqual(receipt["counters"]["nodes"], 7)

    def test_no_project_imports_or_execution_calls(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(item.name.split(".")[0] for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module.split(".")[0])
        self.assertLessEqual(set(imports), {"__future__", "argparse", "ctypes", "hashlib",
                            "json", "math", "os", "sys", "time", "dataclasses",
                            "fractions", "pathlib", "resource"})
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"eval", "exec", "__import__"})


    def test_candidate_row_fields_decode_reading_refs_and_local_alternatives(self):
        payload = synthetic_payload()
        rows = payload["same_axis_candidates"]["candidates"]
        # Producer order: left ref/alt then right ref/alt.
        self.assertEqual(rows[0][3:7], [1, 0, 6, 1])
        frames, yaw, groups, candidates = checker.validate(payload, budget())
        self.assertEqual((frames, yaw), (SYNTHETIC_FRAMES, SYNTHETIC_QUADRANTS))
        first = candidates[0][0]
        self.assertEqual(first.left, ("image-0000", 101))
        self.assertEqual(first.right, ("image-0001", 106))
        self.assertEqual(first.left_b, 2.0)      # ref 1, local alternative 0
        self.assertEqual(first.right_b, 8.0)     # ref 6, local alternative 1
        self.assertEqual(first.axis, 1)
        self.assertEqual(first.delta, Fraction("1.5") - Fraction("6.25"))
        # Every group's kept row stays bound to the group frames and axis.
        for gid, group in enumerate(groups):
            kept = next(item for item in candidates[gid] if item.disposition == "kept")
            self.assertEqual(kept.left[0], group[1])
            self.assertEqual(kept.right[0], group[2])
            self.assertEqual(kept.axis, checker.AXIS_TABLE[
                SYNTHETIC_QUADRANTS[SYNTHETIC_FRAMES.index(group[1])]][group[3]][0])
        self.assertEqual(sum(len(items) for items in candidates), 437)

    def test_candidate_reading_reference_must_be_a_reading_ref(self):
        for field, value in ((3, 18), (3, -1), (5, 18), (5, -1)):
            payload = synthetic_payload()
            payload["same_axis_candidates"]["candidates"][0][field] = value
            with self.assertRaises(ValueError) as caught:
                checker.validate(payload, budget())
            self.assertEqual(str(caught.exception), "invalid_reading_ref")

    def test_candidate_alternative_index_is_local_and_bounded(self):
        for field in (4, 6):
            payload = synthetic_payload()
            payload["same_axis_candidates"]["candidates"][0][field] = 2
            with self.assertRaises(ValueError) as caught:
                checker.validate(payload, budget())
            self.assertEqual(str(caught.exception), "invalid_alternative_ref")

    def test_candidate_frame_and_axis_binding_is_strict(self):
        payload = synthetic_payload()
        # Reading ref 5 belongs to image-0001, not this group's left frame.
        payload["same_axis_candidates"]["candidates"][0][3] = 5
        with self.assertRaises(ValueError) as caught:
            checker.validate(payload, budget())
        self.assertEqual(str(caught.exception), "candidate_group_mismatch")
        payload = synthetic_payload()
        # Same frame as group 3's left endpoint, but local axis 0 != group axis 1.
        rows = payload["same_axis_candidates"]["candidates"]
        kept = next(row for row in rows if row[0] == 3 and row[17] == "kept")
        kept[3] = 5
        with self.assertRaises(ValueError) as caught:
            checker.validate(payload, budget())
        self.assertEqual(str(caught.exception), "candidate_group_mismatch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
