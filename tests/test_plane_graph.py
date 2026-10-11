"""DS-29 synthetic controls for plane identity, quadrant and translation.

The component controls build readings and matches from an exactly known lattice
room (integer eyes, integer faces, known quadrant per frame), so the integer test
is checked against ground truth rather than against another estimator. The
repetition and correlation controls build signatures directly, which is the only
way to isolate the cue from proposal recall. A final image-level control runs
the full reading pipeline and accepts *either* a solved component or a typed
abstention, so proposal failure is reported honestly and no threshold is
weakened.
"""
from __future__ import annotations

import ast
import json
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
import numpy as np
import pytest
import time
import tracemalloc
import warnings

from mc_photogrammetry.pipeline.connected_camera import solve_pair
from mc_photogrammetry.pipeline.contracts import (
    CameraEstimate,
    CameraIntrinsics,
    CameraPose,
)
from mc_photogrammetry.reconstruction import lattice, lattice_planes, plane_graph
from mc_photogrammetry.reconstruction.region_growth import propose_image_planes_v2
from mc_photogrammetry.reconstruction.calibration import CalibrationResult
from mc_photogrammetry.reconstruction.camera import (
    CameraModel,
    camera_rotation_matrix,
    intrinsics_for,
    project_points,
)


def _observe_endpoint_solve(readings, matches, **kwargs):
    observer = plane_graph._EndpointConflictObserver()
    with plane_graph._endpoint_conflict_context(observer):
        solution = plane_graph.solve_component(readings, matches, **kwargs)
    return solution, observer.record()


@pytest.mark.parametrize("same_pair", [True, False])
def test_endpoint_witness_first_conflict_and_canonical_bindings(same_pair):
    readings = [make_reading(f, 1, 1, [(1., 8., 1.), (1./16, .5, 1.)])
                for f in ("a", "b", "c")]
    first = make_match("a", "b", 1, 1, 0, 0, 0)
    second = make_match("a", "b" if same_pair else "c", 1, 1, 0, 0, 0,
                        left_b=1./16)
    matches = (first, second)
    kwargs = {"quadrant_constraints": (("a", "b", 0), ("a", "c", 0))}
    off = plane_graph.solve_component(readings, matches, **kwargs)
    on, record = _observe_endpoint_solve(readings, matches, **kwargs)
    assert _component_bytes(off) == _component_bytes(on)
    assert on.reason == "endpoint_period_conflict"
    assert record["status"] == "complete"
    assert record["planned_assignments"] == record["visited_assignments"] == 1
    assert record["conflicted_assignments"] == 1 and record["evaluator_calls"] == 0
    row = record["witnesses"][0]["first_conflict"]
    category = ("same_endpoint_pair_competing_hypotheses" if same_pair else
                "different_incident_edges")
    assert row["conflict_class"] == category
    assert record["conflict_classes"][category] == 1
    assert row["endpoint"] == ["a", 0]
    assert [row[k]["input_match_index"] for k in ("first", "second")] == [0, 1]
    assert [row[k]["canonical_alternative_index"] for k in ("first", "second")] == [0, 1]
    assert [row[k]["requested_b"] for k in ("first", "second")] == [1., 1./16]
    assert [row[k]["blocks_per_period"] for k in ("first", "second")] == [1., 1./16]
    assert [row[k]["plane_distance_blocks"] for k in ("first", "second")] == [8., .5]
    assert row["first"]["match_identity"] == list(plane_graph._match_key(first))
    assert row["second"]["match_identity"] == list(plane_graph._match_key(second))
    assert row["source_binding"] == record["source_bindings"][0]
    assert len(row["source_binding"]["reading_order_key_sha256"]) == 64
    assert row["source_binding"]["source_region_id"] is None  # no invented provenance


def test_endpoint_observer_same_canonical_alternative_and_inactive_quadrant():
    readings, matches, _ = lattice_component(
        quadrants={"v0": 0, "v1": 0, "v2": 0}, decoy=True)
    constraints = (("v0", "v1", 0), ("v0", "v2", 0))
    compatible = replace(matches[0], left_b=matches[0].left_b + 5e-10)
    inactive = replace(matches[0], left_b=1./16, quadrant_of_right=2)
    supplied = (*matches, compatible, inactive)
    off = plane_graph.solve_component(readings, supplied, quadrant_constraints=constraints)
    on, record = _observe_endpoint_solve(readings, supplied, quadrant_constraints=constraints)
    assert _component_bytes(off) == _component_bytes(on)
    assert record["status"] == "complete" and record["conflicted_assignments"] == 0
    assert record["evaluator_calls"] == 1 and record["witnesses"] == []


def test_endpoint_observer_consistent_three_axes_order_and_numerical_calls(monkeypatch):
    readings, matches, _ = lattice_component(decoy=True)
    original_eval = plane_graph._evaluate_assignment
    original_alternatives = lattice_planes.PlaneReading.matching_alternatives
    order, numerical = [], []
    def evaluated(frames, quadrants, edges, index):
        order.append(tuple(quadrants[f] for f in frames))
        assert edges is matches
        return original_eval(frames, quadrants, edges, index)
    def resolved(reading, b):
        numerical.append((reading.frame, reading.axis, b))
        return original_alternatives(reading, b)
    monkeypatch.setattr(plane_graph, "_evaluate_assignment", evaluated)
    monkeypatch.setattr(lattice_planes.PlaneReading, "matching_alternatives", resolved)
    off = plane_graph.solve_component(readings, matches)
    off_order, off_numerical = list(order), list(numerical)
    order.clear(); numerical.clear()
    on, record = _observe_endpoint_solve(readings, matches)
    assert off.reason == on.reason == ""
    assert _component_bytes(off) == _component_bytes(on)
    assert order == off_order and numerical == off_numerical
    assert record["status"] == "complete"
    assert record["evaluator_calls"] == len(order) == 16
    assert record["planned_assignments"] == record["visited_assignments"] == 16
    assert record["conflicted_assignments"] == 0


def test_endpoint_observer_six_view_planner_256_and_bounded_truncation():
    frames = tuple(f"image-{i:04d}" for i in range(6))
    readings = [make_reading(f, 1, 1, [(1., 8., 1.), (1./16, .5, 1.)]) for f in frames]
    matches = []
    for q in range(4):
        matches += [make_match(frames[0], frames[1], 1, 1, q, 0, 0),
                    make_match(frames[0], frames[1], 1, 1, q, 0, 0, left_b=1./16)]
    constraints = ((frames[3], frames[5], 1),)
    solution, record = _observe_endpoint_solve(readings, matches, quadrant_constraints=constraints)
    assert plane_graph.MAX_ASSIGNMENTS == 16_384
    assert solution.reason == "endpoint_period_conflict"
    assert record["status"] == "complete"
    assert record["planned_assignments"] == record["visited_assignments"] == 256
    assert record["conflicted_assignments"] == record["hashed_assignments"] == 256
    assert record["evaluator_calls"] == 0
    assert record["retained_witness_count"] == record["omitted_witness_count"] == 128
    assert [r["assignment_index"] for r in record["witnesses"]] == list(range(128))
    for row in record["witnesses"]:
        assert (row["quadrants"][5] - row["quadrants"][3]) % 4 == 1
    _, replay = _observe_endpoint_solve(readings, matches, quadrant_constraints=constraints)
    assert replay == record
    # Example retention cannot change the complete ordered digest/counters.
    observer = plane_graph._EndpointConflictObserver()
    observer.MAX_WITNESSES = 0
    with plane_graph._endpoint_conflict_context(observer):
        plane_graph.solve_component(readings, matches, quadrant_constraints=constraints)
    no_examples = observer.record()
    assert no_examples["assignment_sequence_sha256"] == record["assignment_sequence_sha256"]
    assert no_examples["omitted_witness_count"] == 256


def test_endpoint_observer_resource_refusal_preserves_solve_and_missing_hook(monkeypatch):
    readings, matches, _ = lattice_component()
    off = plane_graph.solve_component(readings, matches)
    observer = plane_graph._EndpointConflictObserver()
    observer.MAX_BYTES = 1
    with plane_graph._endpoint_conflict_context(observer):
        on = plane_graph.solve_component(readings, matches)
    record = observer.record()
    assert _component_bytes(off) == _component_bytes(on)
    assert record["status"] == "refused" and record["reason"] == "scalar_byte_budget"
    assert record["visited_assignments"] == record["planned_assignments"] == 16
    assert record["evaluator_calls"] == 16 and record["hashed_assignments"] == 0
    assert plane_graph._EndpointConflictObserver().record()["status"] == "incomplete"
    monkeypatch.setattr(plane_graph, "_assignment_endpoint_period_conflict", lambda *args: True)
    _, missing = _observe_endpoint_solve(readings, matches)
    assert missing["status"] == "refused" and missing["reason"] == "missing_conflict_hook"


def test_endpoint_observer_prebounds_source_rows_and_binding_count(monkeypatch):
    observer = plane_graph._EndpointConflictObserver()
    monkeypatch.setattr(json.JSONEncoder, "iterencode",
                        lambda *_args, **_kwargs: pytest.fail("oversize row reached JSON allocation"))
    assert observer._encode("x" * (observer.MAX_ROW_BYTES + 1)) is None
    assert observer.refusal == "scalar_byte_budget" and observer.work_bytes == 0
    readings, matches, _ = lattice_component()
    observer = plane_graph._EndpointConflictObserver()
    observer.MAX_READINGS = 1
    with plane_graph._endpoint_conflict_context(observer):
        result = plane_graph.solve_component(readings, matches)
    assert result.reason == ""
    assert observer.record()["reason"] == "source_binding_budget"
    assert observer.work_bytes == 0 and observer.bindings == {}

WIDTH, HEIGHT = 720, 480

# Independent quadrant tables (not read from the module): label -> (world axis,
# world sign) for each of the four 90-degree relabellings.
LABELS = {
    (0, 0): 0, (1, 0): 1, (2, 0): 2,
    (0, 1): 2, (1, 1): 1, (2, 1): 0,
    (0, 2): 0, (1, 2): 1, (2, 2): 2,
    (0, 3): 2, (1, 3): 1, (2, 3): 0,
}
SIGNS = {
    (0, 0): 1.0, (1, 0): 1.0, (2, 0): 1.0,
    (0, 1): 1.0, (1, 1): 1.0, (2, 1): -1.0,
    (0, 2): -1.0, (1, 2): 1.0, (2, 2): -1.0,
    (0, 3): -1.0, (1, 3): 1.0, (2, 3): 1.0,
}

TRANSPARENT = lattice_planes.PlaneRectification(
    axis_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    focal=500.0, principal_point=(120.0, 90.0), centre=(0.0, 0.0),
)


def make_reading(frame, axis, side, distances, *, share=0.6, period=60.0):
    alternatives = tuple(
        lattice_planes.DistanceAlternative(float(b), float(d), float(support))
        for b, d, support in distances)
    return lattice_planes.PlaneReading(
        frame=frame, axis=int(axis), side=float(side), alternatives=alternatives,
        period_pixels=float(period), phase_pixels=(0.0, 0.0), extent=1.0, samples=512,
        polygon=None, coverage_cells=8, structural_share=float(share), confidence=0.9,
        rectification=TRANSPARENT)


def make_signature(reading, cells, *, b=None, coverage=None):
    cells = np.asarray(cells, dtype=np.float32)
    if coverage is None:
        coverage = np.ones(cells.shape, dtype=bool)
    chosen_b = reading.alternatives[0].blocks_per_period if b is None else float(b)
    return lattice_planes.PlaneSignature(
        reading=reading, blocks_per_period=chosen_b, cells=cells,
        interior=np.zeros_like(cells), coverage=np.asarray(coverage, dtype=bool),
        origin_uv=(0.0, 0.0))


def make_match(left, right, left_axis, right_axis, quadrant, left_plane, right_plane,
               *, left_b=1.0, right_b=1.0, score=1.0):
    return plane_graph.PlaneMatch(
        left=left, right=right, left_axis=left_axis, right_axis=right_axis,
        left_b=left_b, right_b=right_b, quadrant_of_right=quadrant,
        psr=5.0, peak=1.0, second_peak=0.1, overlap_cells=64,
        integer_residual=None, score=score, left_plane=left_plane, right_plane=right_plane)


def lattice_component(quadrants=None, planes=((0, 0), (1, 0), (2, 0), (2, 20)),
                      eyes=None, decoy=False):
    """Readings and exact matches for an integer lattice room."""
    eyes = {"v0": (5, 2, 5), "v1": (6, 3, 7), "v2": (4, 1, 9)} if eyes is None else eyes
    quadrants = {"v0": 0, "v1": 1, "v2": 0} if quadrants is None else quadrants
    frames = sorted(eyes)
    built = []
    for frame in frames:
        quadrant = quadrants[frame]
        for world_axis, coord in planes:
            label = LABELS[(world_axis, quadrant)]
            world_side = 1.0 if coord > eyes[frame][world_axis] else -1.0
            distance = abs(eyes[frame][world_axis] - coord)
            if distance <= 0.0:
                continue
            side = world_side * SIGNS[(label, quadrant)]
            options = [(1.0, distance, 1.0)]
            if decoy:
                options.append((1.0 / 16.0, distance / 16.0, 1.0))
            reading = make_reading(frame, label, side, options)
            built.append((frame, world_axis, coord, reading))
    readings = [entry[3] for entry in built]
    # Index readings exactly as solve_component does: per frame, by content key.
    index: dict[tuple[str, int, int], int] = {}
    for frame in frames:
        ordered = sorted((entry for entry in built if entry[0] == frame),
                         key=lambda entry: lattice_planes.reading_order_key(entry[3]))
        for position, (_frame, world_axis, coord, _reading) in enumerate(ordered):
            index[(frame, world_axis, coord)] = position
    reading_of = {(entry[0], entry[1], entry[2]): entry[3] for entry in built}
    matches = []
    for first_position, first in enumerate(frames):
        for second in frames[first_position + 1:]:
            quadrant = (quadrants[second] - quadrants[first]) % 4
            for world_axis, coord in planes:
                if (first, world_axis, coord) not in index or (second, world_axis, coord) not in index:
                    continue
                matches.append(make_match(
                    first, second, LABELS[(world_axis, quadrants[first])],
                    LABELS[(world_axis, quadrants[second])], quadrant,
                    index[(first, world_axis, coord)], index[(second, world_axis, coord)],
                    left_b=1.0, right_b=1.0))
    assert reading_of  # keep the mapping meaningful for readers
    return readings, matches, eyes


def _eight_view_component():
    frames = [f"f{index}" for index in range(8)]
    quadrants = dict(zip(frames, (0, 1, 2, 3, 1, 3, 2, 0)))
    eyes = {frame: (5 + index % 3, 2 + index % 2, 5 + index % 4)
            for index, frame in enumerate(frames)}
    readings, complete_matches, found_eyes = lattice_component(
        quadrants=quadrants, planes=((0, 0), (1, 0), (2, 0)), eyes=eyes)
    # A star gives every non-anchor view exactly one hypothesis. The expected
    # assignment is the sole one that activates all seven edges; alternatives
    # leave at least one leaf unresolved and incur the existing penalty.
    axis_for_leaf = {f"f{index}": (index - 1) % 3 for index in range(1, 8)}
    matches = [match for match in complete_matches
               if match.left == "f0" and match.right in axis_for_leaf
               and match.left_axis == axis_for_leaf[match.right]]
    assert len(matches) == 7 and {match.left_axis for match in matches} == {0, 1, 2}
    return readings, matches, found_eyes


def _component_bytes(solution):
    """Canonical public-field encoding used for exact deterministic controls."""
    payload = {
        "frames": list(solution.frames),
        "quadrants": sorted(solution.quadrants.items()),
        "translations": [(key, solution.translations[key]) for key in sorted(solution.translations)],
        "axis_residuals": sorted(solution.axis_residuals.items()),
        "axis_resultants": sorted(solution.axis_resultants.items()),
        "scale_known": solution.scale_known,
        "blocks_per_period": [
            (frame, plane, value)
            for (frame, plane), value in sorted(solution.blocks_per_period.items())
        ],
        "reason": solution.reason,
        "alternatives": [
            (value.quadrants, value.cost, value.resultants, value.axis_observations)
            for value in solution.alternatives
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def test_s1_three_view_component_recovers_quadrants_and_translations():
    readings, matches, eyes = lattice_component()
    solution = plane_graph.solve_component(readings, matches)
    assert solution.reason == ""
    assert solution.scale_known is True
    assert dict(solution.quadrants) == {"v0": 0, "v1": 1, "v2": 0}
    anchor = (5.0, 2.0, 5.0)
    for frame, eye in eyes.items():
        expected = tuple(eye[axis] - anchor[axis] for axis in range(3))
        observed = solution.translations[frame]
        assert max(abs(observed[axis] - expected[axis]) for axis in range(3)) <= 0.25
    assert set(solution.axis_resultants) == {0, 1, 2}
    assert all(value > 0.999 for value in solution.axis_resultants.values())
    annotated = plane_graph.annotate_integer_residuals(solution, readings, matches)
    assert all(match.integer_residual is not None and match.integer_residual < 1e-9
               for match in annotated)
    shifted = [replace(match, cell_shift_rc=((0, 0) if index == 0 else (-8, 8)))
               for index, match in enumerate(matches)]
    shifted_annotated = plane_graph.annotate_integer_residuals(
        solution, readings, shifted)
    assert [match.cell_shift_rc for match in shifted_annotated] == [
        match.cell_shift_rc for match in shifted]


def test_s4_single_plane_pair_resolves_only_its_normal_axis():
    readings, matches, _eyes = lattice_component(
        planes=((0, 0),), eyes={"v0": (5, 2, 5), "v1": (6, 3, 7)}, quadrants={"v0": 0, "v1": 0})
    solution = plane_graph.solve_component(readings, matches)
    assert solution.reason in plane_graph.COMPONENT_ABSTENTION_REASONS
    assert solution.reason == "insufficient_axis_coverage"
    assert set(solution.axis_resultants) == {0}
    assert solution.axis_resultants[0] > 0.999
    assert solution.scale_known is False


def test_s5_two_views_sharing_only_a_floor_leave_the_quadrant_unresolved():
    eye = {"v0": (5.0, 2.0, 5.0), "v1": (6.0, 3.0, 7.0)}
    base = make_reading("v0", 1, -1.0, [(1.0, 2.0, 1.0)])
    other = make_reading("v1", 1, -1.0, [(1.0, 3.0, 1.0)])
    # A floor is the same surface under both the parity-0 and parity-2 quadrant
    # hypotheses, so the matcher would emit both; the integer test cannot choose.
    matches = [
        make_match("v0", "v1", 1, 1, 0, 0, 0),
        make_match("v0", "v1", 1, 1, 2, 0, 0),
    ]
    solution = plane_graph.solve_component([base, other], matches)
    assert solution.reason == "ambiguous_quadrant"
    assert solution.scale_known is False


def test_solved_camera_quadrant_constraint_removes_the_floor_tie():
    base = make_reading("v0", 1, -1.0, [(1.0, 2.0, 1.0)])
    other = make_reading("v1", 1, -1.0, [(1.0, 3.0, 1.0)])
    matches = [
        make_match("v0", "v1", 1, 1, 0, 0, 0),
        make_match("v0", "v1", 1, 1, 2, 0, 0),
    ]
    solution = plane_graph.solve_component(
        [base, other], matches,
        quadrant_constraints=(("v0", "v1", 0),))
    assert solution.reason == "insufficient_axis_coverage"
    assert dict(solution.quadrants) == {"v0": 0, "v1": 0}
    assert all(item.quadrants == (0, 0) for item in solution.alternatives)


def test_incomplete_camera_constraints_preserve_remaining_ambiguity():
    frames = ("v0", "v1", "v2")
    readings = [make_reading(frame, 1, -1.0, [(1.0, 2.0, 1.0)])
                for frame in frames]
    matches = []
    for left, right in zip(frames, frames[1:]):
        matches.extend(make_match(left, right, 1, 1, quadrant, 0, 0)
                       for quadrant in (0, 2))
    solution = plane_graph.solve_component(
        readings, matches, quadrant_constraints=(("v0", "v1", 0),))
    assert solution.reason == "ambiguous_quadrant"
    assert len({item.quadrants[2] for item in solution.alternatives}) > 1


def test_malformed_foreign_and_contradictory_quadrant_constraints_are_atomic():
    base = make_reading("v0", 1, -1.0, [(1.0, 2.0, 1.0)])
    other = make_reading("v1", 1, -1.0, [(1.0, 3.0, 1.0)])
    matches = (make_match("v0", "v1", 1, 1, 0, 0, 0),)
    with pytest.raises(ValueError, match="immutable tuple"):
        plane_graph.solve_component([base, other], matches,
                                    quadrant_constraints=[])
    with pytest.raises(ValueError, match="foreign frame"):
        plane_graph.solve_component(
            [base, other], matches,
            quadrant_constraints=(("v0", "foreign", 0),))
    contradictory = plane_graph.solve_component(
        [base, other], matches,
        quadrant_constraints=(("v0", "v1", 0), ("v1", "v0", 2)))
    assert contradictory.reason == "quadrant_constraint_inconsistent"
    assert contradictory.alternatives == () and contradictory.scale_known is False


def _brute_quadrant_plan(frames, constraints):
    """Small reference: the original full-frame Cartesian order and filter."""
    accepted = []
    for tail in product(range(4), repeat=len(frames) - 1):
        assignment = dict(zip(frames, (0, *tail)))
        if all((assignment[right] - assignment[left]) % 4 == relative
               for (left, right), relative in constraints.items()):
            accepted.append(assignment)
    return ("", len(accepted), iter(accepted)) if accepted else (
        "quadrant_constraint_inconsistent", 0, iter(()))


@pytest.mark.parametrize("edges", (
    (), (("a", "d", 3),),
    (("a", "d", 3), ("b", "e", 2)),
    (("a", "c", 3), ("c", "e", 2), ("a", "e", 1)),
    (("a", "c", 3), ("c", "e", 2), ("a", "e", 0)),
))
def test_constraint_planner_matches_brute_order_with_isolates_and_wrap(edges):
    frames = tuple("abcde")
    constraints = {(left, right): relative for left, right, relative in edges}
    reason, count, assignments = plane_graph._quadrant_assignment_plan(
        frames, constraints)
    expected_reason, expected_count, reference = _brute_quadrant_plan(
        frames, constraints)
    assert reason == expected_reason and count == expected_count
    assert iter(assignments) is assignments  # no materialized assignment set
    assert list(assignments) == list(reference)
    # Edge insertion order cannot affect roots, offsets, or enumeration.
    reverse = plane_graph._quadrant_assignment_plan(
        frames, dict(reversed(tuple(constraints.items()))))
    assert reverse[:2] == (reason, count)
    assert list(reverse[2]) == list(_brute_quadrant_plan(frames, constraints)[2])


@pytest.mark.parametrize("edges", (
    (), (("v0", "v1", 0),),
    (("v1", "v0", 0), ("v0", "v1", 0)),
    (("v0", "v1", 0), ("v1", "v2", 2), ("v2", "v0", 2)),
    (("v0", "v1", 0), ("v1", "v2", 2), ("v2", "v0", 1)),
))
@pytest.mark.parametrize("fixture", ("floor_ties", "three_axes"))
def test_complete_solver_result_equals_brute_reference(monkeypatch, edges, fixture):
    if fixture == "floor_ties":
        frames = ("v0", "v1", "v2")
        readings = [make_reading(frame, 1, -1, [(1, 2, 1)]) for frame in frames]
        matches = [make_match(left, right, 1, 1, quadrant, 0, 0)
                   for left, right in zip(frames, frames[1:])
                   for quadrant in (0, 2)]
    else:
        readings, matches, _ = lattice_component()
    # Shuffled source order exercises canonical indexing, not just the planner.
    readings, matches = list(reversed(readings)), list(reversed(matches))
    found = plane_graph.solve_component(readings, matches, quadrant_constraints=edges)
    monkeypatch.setattr(plane_graph, "_quadrant_assignment_plan", _brute_quadrant_plan)
    reference = plane_graph.solve_component(readings, matches, quadrant_constraints=edges)
    assert asdict(found) == asdict(reference)
    if fixture == "floor_ties" and edges in ((), (("v0", "v1", 0),)):
        assert found.reason == "ambiguous_quadrant"


@pytest.mark.parametrize("components,expected_calls,reason", (
    (8, 16_384, "quadrant_constraint_inconsistent"),
    (9, 0, "component_too_large"),
))
def test_true_component_budget_counts_full_evaluations(
        monkeypatch, components, expected_calls, reason):
    # One constrained pair plus isolated reading-present frames. Total frames
    # exceed the old frame-count limit even for the accepted eight components.
    frames = tuple(f"f{index:02d}" for index in range(components + 1))
    readings = [make_reading(frame, 1, -1, [(1, 2, 1)]) for frame in frames]
    matches = [make_match(frames[0], frames[1], 1, 1, 0, 0, 0)]
    calls = {"period": 0, "evaluate": 0}
    def period(*_args):
        calls["period"] += 1
        return False
    def evaluate(*_args):
        calls["evaluate"] += 1
        return None
    monkeypatch.setattr(plane_graph, "_assignment_endpoint_period_conflict", period)
    monkeypatch.setattr(plane_graph, "_evaluate_assignment", evaluate)
    solution = plane_graph.solve_component(
        readings, matches, quadrant_constraints=((frames[0], frames[1], 0),))
    assert calls == {"period": expected_calls, "evaluate": expected_calls}
    assert solution.reason == reason and solution.alternatives == ()
    assert plane_graph.MAX_ASSIGNMENTS == 16_384


def test_contradictory_cycle_and_unconstrained_nine_refuse_before_numerics(monkeypatch):
    frames = tuple(f"f{index:02d}" for index in range(9))
    readings = [make_reading(frame, 1, -1, [(1, 2, 1)]) for frame in frames]
    matches = [make_match(frames[0], frames[1], 1, 1, 0, 0, 0)]
    def forbidden(*_args):
        pytest.fail("refused plans must not evaluate assignments")
    monkeypatch.setattr(plane_graph, "_assignment_endpoint_period_conflict", forbidden)
    monkeypatch.setattr(plane_graph, "_evaluate_assignment", forbidden)
    cycle = ((frames[0], frames[1], 1), (frames[1], frames[2], 1),
             (frames[0], frames[2], 3))
    assert plane_graph.solve_component(
        readings, matches, quadrant_constraints=cycle).reason == "quadrant_constraint_inconsistent"
    assert plane_graph.solve_component(readings, matches).reason == "component_too_large"


def test_fully_constrained_32_frames_solve_all_three_axes_once(monkeypatch):
    frames = tuple(f"f{index:02d}" for index in range(32))
    quadrants = {frame: index % 4 for index, frame in enumerate(frames)}
    eyes = {frame: (5 + index % 3, 2 + index % 2, 5 + index % 4)
            for index, frame in enumerate(frames)}
    readings, complete, _ = lattice_component(
        quadrants=quadrants, eyes=eyes, planes=((0, 0), (1, 0), (2, 0)))
    matches = [match for match in complete if match.left == frames[0]]
    constraints = tuple((frame, frames[0], (-quadrants[frame]) % 4)
                        for frame in reversed(frames[1:]))
    constraints += ((frames[0], frames[1], quadrants[frames[1]]),)  # agreeing duplicate
    original = plane_graph._evaluate_assignment
    calls = []
    def evaluate(*args):
        calls.append(dict(args[1]))
        return original(*args)
    monkeypatch.setattr(plane_graph, "_evaluate_assignment", evaluate)
    solution = plane_graph.solve_component(
        list(reversed(readings)), list(reversed(matches)), quadrant_constraints=constraints)
    assert calls == [quadrants]
    assert solution.reason == "" and solution.scale_known
    assert dict(solution.quadrants) == quadrants
    assert len(solution.alternatives) == 1 and set(solution.axis_resultants) == {0, 1, 2}
    for frame in frames:
        assert solution.translations[frame] == pytest.approx(
            np.asarray(eyes[frame]) - np.asarray(eyes[frames[0]]), abs=1e-10)
    assert min(solution.axis_resultants.values()) > .999999


def test_wrong_b_is_rejected_by_the_integer_test():
    readings, matches, eyes = lattice_component(decoy=True)
    # Replace every match's right b with the texel decoy: the faces no longer
    # close on integers, so the component must abstain rather than solve.
    wrong = [
        make_match(match.left, match.right, match.left_axis, match.right_axis,
                   match.quadrant_of_right, match.left_plane, match.right_plane,
                   left_b=1.0 / 16.0, right_b=1.0 / 16.0, score=match.score)
        for match in matches
    ]
    solution = plane_graph.solve_component(readings, wrong)
    correct = plane_graph.solve_component(readings, matches)
    # The wrong reading cannot close on integers: it abstains, and its best
    # assignment cost is strictly worse than the correct reading's.
    assert solution.reason in plane_graph.COMPONENT_ABSTENTION_REASONS
    assert solution.scale_known is False
    assert correct.reason == ""
    assert solution.alternatives[0].cost > correct.alternatives[0].cost
    assert all(solution.axis_resultants.get(axis, 1.0) <= correct.axis_resultants[axis] + 1e-9
               for axis in correct.axis_resultants)


def test_bu_cross_edge_scale_alias_control(monkeypatch):
    """Distinct canonical periods on one active endpoint refuse before solving."""
    gates = {"integer_tolerance": 0.15, "min_resultant": 0.90, "ambiguous_margin": 0.02}
    readings, matches, _eyes = lattice_component(decoy=True)
    constraints = (("v0", "v1", 1), ("v0", "v2", 0), ("v1", "v2", 3))
    baseline = plane_graph.solve_component(
        readings, matches, quadrant_constraints=constraints, **gates)
    assert baseline.reason == ""
    assert baseline.scale_known is True
    assert dict(baseline.quadrants) == {"v0": 0, "v1": 1, "v2": 0}
    assert all(value == pytest.approx(1.0) for value in baseline.axis_resultants.values())

    target = next(match for match in matches
                  if match.left == "v0" and match.right == "v1" and match.left_axis == 0)
    other = next(match for match in matches
                 if match.left == "v0" and match.right == "v2" and match.left_axis == 0)
    # One v0 axis-0 reading is the left endpoint of two different edges.
    assert (target.left, target.left_plane) == ("v0", 0)
    assert (other.left, other.left_plane) == ("v0", 0)
    assert target.right == "v1" and other.right == "v2"
    assert target.left_b == other.left_b == 1.0
    assert plane_graph._match_key(target) != plane_graph._match_key(other)

    # The alternatives are uniquely resolved: the b gap is 0.9375, not noise.
    assert abs((1.0 - 1.0 / 16.0) - 0.9375) < 1e-12
    read_by_index, _index_by_object, _frames = plane_graph._canonical_reading_index(readings)
    endpoint_reading = read_by_index[("v0", target.left_plane)]
    assert endpoint_reading.distance_for(1.0) == pytest.approx(5.0)
    assert endpoint_reading.distance_for(1.0 / 16.0) == pytest.approx(5.0 / 16.0)

    # Exactly one legal alias substitution on the shared endpoint; the other
    # endpoint of the cloned edge stays at b = 1.
    mixed_matches = [replace(match, left_b=1.0 / 16.0) if match is target else match
                     for match in matches]
    assert mixed_matches != matches
    assert len(mixed_matches) == len(matches)
    assert mixed_matches[matches.index(target)].right_b == 1.0
    solver_keys = [(match.left, match.left_plane, match.right, match.right_plane,
                    match.quadrant_of_right) for match in mixed_matches]
    # Different full edge keys, so no edge-local duplicate identity exists.
    assert len(solver_keys) == len(set(solver_keys))

    with monkeypatch.context() as blocked:
        blocked.setattr(plane_graph.np.linalg, "lstsq",
                        lambda *_a, **_k: pytest.fail("conflict reached least squares"))
        blocked.setattr(plane_graph, "_evaluate_assignment",
                        lambda *_a, **_k: pytest.fail("conflict reached evaluator"))
        mixed = plane_graph.solve_component(
            readings, mixed_matches, quadrant_constraints=constraints, **gates)
    assert mixed.reason == "endpoint_period_conflict"
    assert mixed.scale_known is False
    assert mixed.frames == baseline.frames
    # Typed refusals preserve the existing zero-filled public return shape.
    assert mixed.translations == {frame: (0.0, 0.0, 0.0) for frame in baseline.frames}
    assert not mixed.blocks_per_period
    assert not mixed.axis_resultants and not mixed.alternatives

    # Frame-pair-grouped same-axis candidates both remain: the two edges sit in
    # different groups, so the existing filter neither compares nor drops them.
    refusals: list = []
    kept = plane_graph._apply_same_axis_filter(list(mixed_matches), read_by_index, refusals)
    assert refusals == []
    assert len(kept) == len(mixed_matches)
    alias_kept = next(match for match in kept
                      if match.left == "v0" and match.right == "v1" and match.left_axis == 0)
    plain_kept = next(match for match in kept
                      if match.left == "v0" and match.right == "v2" and match.left_axis == 0)
    assert alias_kept.left_b == 1.0 / 16.0
    assert plain_kept.left_b == 1.0
    assert (alias_kept.left, alias_kept.left_plane) == (plain_kept.left, plain_kept.left_plane)

    # Restoring the original match list recovers the exact baseline.
    restored = plane_graph.solve_component(
        readings, matches, quadrant_constraints=constraints, **gates)
    assert restored.reason == ""
    assert _component_bytes(restored) == _component_bytes(baseline)


def test_endpoint_period_conflict_does_not_block_consistent_quadrant(monkeypatch):
    readings, original, _eyes = lattice_component(
        quadrants={"v0": 0, "v1": 0, "v2": 0},
        planes=((0, 0), (1, 0), (2, 0)), decoy=True)
    target = next(match for match in original
                  if match.left == "v0" and match.right == "v1" and match.left_axis == 0)
    matches = [replace(match, left_b=1.0 / 16.0) if match is target else match
               for match in original]
    matches += [replace(match, quadrant_of_right=2) for match in original
                if "v1" in (match.left, match.right)]
    supplied = tuple(matches)
    evaluated = []
    evaluate = plane_graph._evaluate_assignment

    def observe(frames, quadrants, edges, index):
        assert tuple(edges) == supplied  # no partition or edge dropping
        evaluated.append(dict(quadrants))
        return evaluate(frames, quadrants, edges, index)

    monkeypatch.setattr(plane_graph, "_evaluate_assignment", observe)
    solution = plane_graph.solve_component(
        readings, matches, quadrant_constraints=(("v0", "v2", 0),))
    assert {"v0": 0, "v1": 0, "v2": 0} not in evaluated
    assert {"v0": 0, "v1": 2, "v2": 0} in evaluated
    assert solution.reason == "" and solution.scale_known
    assert dict(solution.quadrants) == {"v0": 0, "v1": 2, "v2": 0}
    assert solution.frames == ("v0", "v1", "v2")
    assert set(solution.translations) == set(solution.frames)
    assert tuple(matches) == supplied
    # All nine compatible edges contribute both endpoints to the phase test.
    assert solution.alternatives[0].axis_observations == (6, 6, 6)


def test_inactive_endpoint_period_aliases_do_not_conflict():
    readings, matches, _eyes = lattice_component(decoy=True)
    constraints = (("v0", "v1", 1), ("v0", "v2", 0))
    baseline = plane_graph.solve_component(
        readings, matches, quadrant_constraints=constraints)
    inactive = replace(matches[0], quadrant_of_right=(matches[0].quadrant_of_right + 2) % 4,
                       left_b=1.0 / 16.0)
    augmented = [*matches, inactive]
    solution = plane_graph.solve_component(
        readings, augmented, quadrant_constraints=constraints)
    assert _component_bytes(solution) == _component_bytes(baseline)
    assert augmented[-1] is inactive and len(augmented) == len(matches) + 1


@pytest.mark.parametrize("conflicting", (False, True))
def test_endpoint_period_consistency_is_permutation_and_reversal_stable(conflicting):
    readings, matches, _eyes = lattice_component(decoy=True)
    constraints = (("v0", "v1", 1), ("v0", "v2", 0))
    if conflicting:
        target = next(match for match in matches
                      if match.left == "v0" and match.right == "v1" and match.left_axis == 0)
        matches = [replace(match, left_b=1.0 / 16.0) if match is target else match
                   for match in matches]
    baseline = plane_graph.solve_component(
        readings, matches, quadrant_constraints=constraints)
    reversed_endpoints = [replace(
        match, left=match.right, right=match.left,
        left_axis=match.right_axis, right_axis=match.left_axis,
        left_plane=match.right_plane, right_plane=match.left_plane,
        left_b=match.right_b, right_b=match.left_b,
        quadrant_of_right=(-match.quadrant_of_right) % 4) for match in matches]
    for edges in (list(reversed(matches)), list(reversed(reversed_endpoints))):
        supplied = tuple(edges)
        solution = plane_graph.solve_component(
            list(reversed(readings)), edges, quadrant_constraints=constraints)
        assert solution.reason == baseline.reason == (
            "endpoint_period_conflict" if conflicting else "")
        assert solution.frames == baseline.frames
        assert dict(solution.quadrants) == dict(baseline.quadrants)
        assert dict(solution.blocks_per_period) == dict(baseline.blocks_per_period)
        assert solution.scale_known == baseline.scale_known
        for frame in baseline.translations:
            assert solution.translations[frame] == pytest.approx(baseline.translations[frame])
        assert tuple(edges) == supplied and len(edges) == len(matches)
        if not conflicting:
            # Repeated identical endpoint alternatives across different edges
            # remain valid, and every edge remains an observation.
            assert sum(solution.alternatives[0].axis_observations) == 2 * len(edges)


def test_requested_b_noise_resolves_to_same_canonical_endpoint_alternative():
    readings, matches, _eyes = lattice_component(decoy=True)
    target = next(match for match in matches
                  if match.left == "v0" and match.right == "v1" and match.left_axis == 0)
    noisy = [replace(match, left_b=1.0 + 5e-10) if match is target else match
             for match in matches]
    solution = plane_graph.solve_component(
        readings, noisy, quadrant_constraints=(("v0", "v1", 1), ("v0", "v2", 0)))
    assert solution.reason == "" and solution.scale_known
    assert solution.frames == ("v0", "v1", "v2")
    assert sum(solution.alternatives[0].axis_observations) == 2 * len(noisy)


def test_endpoint_period_guard_preserves_production_connectivity_refusal(monkeypatch):
    from mc_photogrammetry.pipeline import image_derived_graph

    readings, complete, _eyes = lattice_component(
        quadrants={"v0": 0, "v1": 0, "v2": 0}, planes=((0, 0), (1, 0), (2, 0)))
    constraints = (("v0", "v1", 0), ("v0", "v2", 0))
    baseline = plane_graph.solve_component(
        readings, complete, quadrant_constraints=constraints)
    assert image_derived_graph._production_solution_refusal(
        baseline.frames, readings, complete, baseline) is None
    # Synthetic input deliberately lacks two normal-axis connections for v2.
    sparse = [match for match in complete
              if match.left_axis == 0 or "v2" not in (match.left, match.right)]
    supplied = tuple(sparse)
    evaluate = plane_graph._evaluate_assignment

    def observe(frames, quadrants, edges, index):
        assert tuple(edges) == supplied
        assert frames == baseline.frames
        return evaluate(frames, quadrants, edges, index)

    monkeypatch.setattr(plane_graph, "_evaluate_assignment", observe)
    solution = plane_graph.solve_component(
        readings, sparse, quadrant_constraints=constraints)
    assert solution.reason == ""  # existing permissive standalone solver
    assert solution.frames == baseline.frames
    assert set(solution.translations) == set(baseline.frames)
    assert tuple(sparse) == supplied
    assert image_derived_graph._production_solution_refusal(
        solution.frames, readings, sparse, solution) == (
            "graph_translation_unobservable", "world axis 1 is disconnected")


def test_s7_frame_relabelling_and_input_order_do_not_change_the_solution():
    readings, matches, _eyes = lattice_component()
    baseline = plane_graph.solve_component(readings, matches)

    shuffled = list(readings)
    np.random.default_rng(29).shuffle(shuffled)
    permuted_matches = list(matches)
    np.random.default_rng(30).shuffle(permuted_matches)
    reordered = plane_graph.solve_component(shuffled, permuted_matches)
    assert reordered.reason == ""
    assert dict(reordered.quadrants) == dict(baseline.quadrants)
    for frame in baseline.frames:
        assert reordered.translations[frame] == pytest.approx(baseline.translations[frame])

    rename = {"v0": "alpha", "v1": "beta", "v2": "gamma"}
    renamed_readings = [
        make_reading(rename[reading.frame], reading.axis, reading.side,
                     [(a.blocks_per_period, a.distance_blocks, a.support)
                      for a in reading.alternatives],
                     share=reading.structural_share, period=reading.period_pixels)
        for reading in readings
    ]
    renamed_matches = [
        make_match(rename[match.left], rename[match.right], match.left_axis, match.right_axis,
                   match.quadrant_of_right, match.left_plane, match.right_plane,
                   left_b=match.left_b, right_b=match.right_b, score=match.score)
        for match in matches
    ]
    relabelled = plane_graph.solve_component(renamed_readings, renamed_matches)
    assert relabelled.reason == ""
    for frame, new_name in rename.items():
        assert relabelled.quadrants[new_name] == baseline.quadrants[frame]
        assert relabelled.translations[new_name] == pytest.approx(baseline.translations[frame])


def test_g1a_expanded_cap_is_field_exactly_equal_to_legacy_cap_for_small_fixtures(monkeypatch):
    ambiguous_readings = [
        make_reading("a", 1, -1.0, [(1.0, 5.0, 1.0)]),
        make_reading("b", 1, -1.0, [(1.0, 5.0, 1.0)]),
    ]
    ambiguous_matches = [
        make_match("a", "b", 1, 1, q, 0, 0) for q in (0, 2)
    ]
    five_frames = [f"w{index}" for index in range(5)]
    five_quadrants = dict(zip(five_frames, (0, 3, 1, 2, 1)))
    five_eyes = {frame: (4 + index % 3, 2 + index % 2, 6 + index % 4)
                 for index, frame in enumerate(five_frames)}
    five_readings, five_matches, _ = lattice_component(
        quadrants=five_quadrants, planes=((0, 0), (1, 0), (2, 0)), eyes=five_eyes)
    correct_readings, correct_matches, _ = lattice_component()
    wrong_b = [make_match(
        match.left, match.right, match.left_axis, match.right_axis,
        match.quadrant_of_right, match.left_plane, match.right_plane,
        left_b=1.0 / 16.0, right_b=1.0 / 16.0, score=match.score)
        for match in correct_matches]
    cases = [
        (correct_readings, correct_matches),
        (correct_readings, wrong_b),
        (ambiguous_readings, ambiguous_matches),
        (five_readings, five_matches),
        ([], []),
    ]
    for readings, matches in cases:
        assert len({reading.frame for reading in readings}) <= 5
        monkeypatch.setattr(plane_graph, "MAX_ASSIGNMENTS", 256)
        legacy = plane_graph.solve_component(readings, matches)
        monkeypatch.setattr(plane_graph, "MAX_ASSIGNMENTS", 16_384)
        expanded = plane_graph.solve_component(readings, matches)
        assert expanded == legacy
        assert _component_bytes(expanded) == _component_bytes(legacy)


def test_g1a_eight_view_unique_solution_is_exact_deterministic_and_bounded(monkeypatch):
    readings, matches, _eyes = _eight_view_component()
    expected_quadrants = {f"f{index}": value for index, value in enumerate((0, 1, 2, 3, 1, 3, 2, 0))}
    original_evaluate = plane_graph._evaluate_assignment
    calls = 0

    def counted_evaluate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(plane_graph, "_evaluate_assignment", counted_evaluate)
    tracemalloc.start()
    started = time.perf_counter()
    try:
        first = plane_graph.solve_component(readings, matches)
        elapsed = time.perf_counter() - started
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert calls == 16_384
    assert elapsed < 120.0
    assert peak <= 128 * 1024 * 1024
    assert first.reason == ""
    assert dict(first.quadrants) == expected_quadrants
    assert first.scale_known is True

    rng = np.random.default_rng(411)
    reordered_readings = list(readings)
    reordered_matches = list(matches)
    rng.shuffle(reordered_readings)
    rng.shuffle(reordered_matches)
    second = plane_graph.solve_component(reordered_readings, reordered_matches)
    assert _component_bytes(first) == _component_bytes(second)


def test_g1a_eight_view_vertical_q_or_q_plus_two_remains_ambiguous():
    frames = [f"v{index}" for index in range(8)]
    readings = [make_reading(frame, 1, -1.0, [(1.0, 5.0, 1.0)]) for frame in frames]
    matches = []
    for left, right in zip(frames, frames[1:]):
        matches.extend(make_match(left, right, 1, 1, q, 0, 0) for q in (0, 2))
    solution = plane_graph.solve_component(readings, matches)
    assert solution.reason == "ambiguous_quadrant"
    assert len(solution.alternatives) == plane_graph.MAX_ALTERNATIVES
    assert solution.alternatives[0].cost == solution.alternatives[1].cost
    assert solution.alternatives[0].quadrants != solution.alternatives[1].quadrants


def test_g1a_nine_view_component_abstains_before_enumeration(monkeypatch):
    readings = [make_reading(f"n{index}", 1, -1.0, [(1.0, 5.0, 1.0)])
                for index in range(9)]
    matches = [make_match("n0", "n1", 1, 1, 0, 0, 0)]
    calls = 0
    original_evaluate = plane_graph._evaluate_assignment

    def counted_evaluate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(plane_graph, "_evaluate_assignment", counted_evaluate)
    solution = plane_graph.solve_component(readings, matches)
    assert solution.reason == "component_too_large"
    assert calls == 0


def test_repetition_gate_refuses_a_uniform_periodic_wall():
    # A uniform repeating wall's signature is nearly constant cell energy: after
    # mean subtraction there is no unique shift, so no identity may be claimed.
    rng = np.random.default_rng(29)
    flat_left = 1.0 + 0.001 * rng.standard_normal((24, 24))
    flat_right = 1.0 + 0.001 * rng.standard_normal((24, 24))
    reading_left = make_reading("a", 0, -1.0, [(1.0, 6.0, 1.0)], share=0.05)
    reading_right = make_reading("b", 0, -1.0, [(1.0, 6.0, 1.0)], share=0.05)
    left = make_signature(reading_left, flat_left, b=1.0)
    right = make_signature(reading_right, flat_right, b=1.0)
    refusals: list = []
    assert plane_graph.match_plane_pair(left, right, quadrant=0, refusals=refusals) is None
    assert any(refusal.reason == "repetition_unresolved" for refusal in refusals)


def test_correlation_finds_a_known_shift_and_scores_a_unique_match():
    rng = np.random.default_rng(29)
    base = rng.random((28, 28))
    shifted = np.roll(base, (3, -4), axis=(0, 1))
    surface = plane_graph._correlation_surface(base, shifted, max_shift_cells=8)
    assert surface is not None
    peak, psr, second, shift = plane_graph._peak_statistics(surface, max_shift_cells=8)
    assert shift == (-3, 4)
    assert psr > 3.0
    assert peak > second

    reading_left = make_reading("a", 0, -1.0, [(1.0, 6.0, 1.0)], share=0.6)
    reading_right = make_reading("b", 0, -1.0, [(1.0, 6.0, 1.0)], share=0.6)
    left = make_signature(reading_left, base, b=1.0)
    right = make_signature(reading_right, shifted, b=1.0)
    match = plane_graph.match_plane_pair(left, right, quadrant=0)
    assert match is not None
    assert match.cell_shift_rc == (-3, 4)
    assert match.psr > 3.0
    assert match.overlap_cells > 0


def test_empty_sidelobe_statistics_are_undefined_without_numpy_warning():
    # Radius 32 with a single admissible cell mirrors the recorded graph-stage
    # geometry: every other shift fails the overlap gate, so the peak's 3x3
    # exclusion window leaves no sidelobe sample at all.
    surface = np.zeros((65, 65), dtype=np.float64)
    surface[32, 32] = 1.0
    admissible = np.zeros((65, 65), dtype=bool)
    admissible[32, 32] = True
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        peak, psr, second, shift = plane_graph._peak_statistics(
            surface, max_shift_cells=32, admissible=admissible)
    assert captured == []
    assert peak == 1.0
    assert psr is None
    assert second == 0.0
    assert shift == (0, 0)


def test_empty_sidelobes_refuse_with_typed_reason_and_json_safe_trace():
    # A zero-radius search has a 1x1 surface, so its only admissible cell is the
    # peak: no sidelobe sample exists. The pair must abstain, and the diagnostic
    # trace must stay JSON-safe for the strict attribution digest.
    rng = np.random.default_rng(20261002)
    cells = rng.random((24, 24))
    left_reading = make_reading("a", 0, -1.0, [(1.0, 6.0, 1.0)], share=0.6)
    right_reading = make_reading("b", 0, -1.0, [(1.0, 6.0, 1.0)], share=0.6)
    left = make_signature(left_reading, cells, b=1.0)
    right = make_signature(right_reading, cells, b=1.0)
    refusals: list = []
    trace = plane_graph.PlaneGraphTrace([])
    assert plane_graph.match_plane_pair(
        left, right, quadrant=0, max_shift_cells=0,
        refusals=refusals, diagnostic_sink=trace) is None
    assert [refusal.reason for refusal in refusals] == ["repetition_unresolved"]
    assert refusals[0].detail == (
        "no admissible sidelobe cells outside the peak: peak-to-sidelobe "
        "ratio is undefined")
    assert len(trace.pairs) == 1
    row = trace.pairs[0]
    assert row.admitted is False
    assert row.psr is None
    for value in (row.raw_max_value, row.supported_max_value, row.psr,
                  row.score, row.overlap_fraction):
        assert value is None or np.isfinite(value)
    encoded = json.dumps(asdict(row), sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
    assert "Infinity" not in encoded and "NaN" not in encoded


def test_empty_sidelobe_refusal_trace_survives_attribution_digest():
    from mc_photogrammetry.pipeline import image_derived_graph_attribution as attribution_mod

    rng = np.random.default_rng(20261003)
    cells = rng.random((24, 24))
    left_reading = make_reading("a", 0, -1.0, [(1.0, 6.0, 1.0)], share=0.6)
    right_reading = make_reading("b", 0, -1.0, [(1.0, 6.0, 1.0)], share=0.6)
    left = make_signature(left_reading, cells, b=1.0)
    right = make_signature(right_reading, cells, b=1.0)
    trace = plane_graph.PlaneGraphTrace([])
    assert plane_graph.match_plane_pair(
        left, right, quadrant=0, max_shift_cells=0,
        diagnostic_sink=trace) is None
    digest = "0123456789abcdef" * 4
    attribution = attribution_mod.ImageDerivedGraphAttribution(
        "mc-photogrammetry-image-derived-graph-attribution-v4",
        plane_graph.PLANE_CORRELATION_ALGORITHM,
        digest, digest, digest, digest, digest,
        (("max_shift_cells", 0),),
        "admission", "geometry_unavailable", "no admissible sidelobe cells",
        trace.pre_admission_candidates, trace.post_admission_candidates,
        trace.post_same_axis_candidates, tuple(trace.pairs), (), (), ())
    assert len(attribution.attribution_sha256) == 64
    replayed = json.dumps(
        attribution.to_record(), sort_keys=True, separators=(",", ":"),
        allow_nan=False)
    assert "Infinity" not in replayed and "NaN" not in replayed


def test_populated_sidelobes_keep_finite_psr_and_admission():
    rng = np.random.default_rng(29)
    base = rng.random((28, 28))
    shifted = np.roll(base, (3, -4), axis=(0, 1))
    surface = plane_graph._correlation_surface(base, shifted, max_shift_cells=8)
    assert surface is not None
    peak, psr, second, shift = plane_graph._peak_statistics(surface, max_shift_cells=8)
    assert psr is not None and np.isfinite(psr) and psr > 3.0
    assert shift == (-3, 4) and peak > second
    reading_left = make_reading("a", 0, -1.0, [(1.0, 6.0, 1.0)], share=0.6)
    reading_right = make_reading("b", 0, -1.0, [(1.0, 6.0, 1.0)], share=0.6)
    left = make_signature(reading_left, base, b=1.0)
    right = make_signature(reading_right, shifted, b=1.0)
    match = plane_graph.match_plane_pair(left, right, quadrant=0)
    assert match is not None
    assert np.isfinite(match.psr) and match.psr > 3.0
    assert match.cell_shift_rc == shift


@pytest.mark.parametrize(
    "shape,roll",
    (((31, 46), (0, 0)), ((31, 46), (3, -4)), ((37, 52), (-5, 7)),
     ((44, 33), (6, 2)), ((45, 60), (8, -8))),
)
def test_correlation_shift_is_signed_row_column_on_asymmetric_non_square_inputs(shape, roll):
    rng = np.random.default_rng(927)
    base = rng.random(shape)
    right = np.roll(base, roll, axis=(0, 1))
    surface = plane_graph._correlation_surface(base, right, max_shift_cells=8)
    assert surface is not None
    _peak, _psr, _second, shift = plane_graph._peak_statistics(
        surface, max_shift_cells=8)
    assert shift == (-roll[0], -roll[1])

    row_start, row_stop = max(0, -shift[0]), min(shape[0], shape[0] - shift[0])
    col_start, col_stop = max(0, -shift[1]), min(shape[1], shape[1] - shift[1])
    rows = slice(row_start, row_stop)
    columns = slice(col_start, col_stop)
    np.testing.assert_allclose(
        base[rows.start + shift[0]:rows.stop + shift[0],
             columns.start + shift[1]:columns.stop + shift[1]],
        right[rows, columns], rtol=0.0, atol=0.0)


def test_match_graph_copies_admitted_cell_shift_and_rejects_malformed_shift():
    reading_left = make_reading("a", 0, -1.0, [(1.0, 6.0, 1.0)], share=0.6)
    reading_right = make_reading("b", 0, -1.0, [(1.0, 6.0, 1.0)], share=0.6)
    rng = np.random.default_rng(721)
    left_cells = rng.random((28, 28))
    right_cells = np.roll(left_cells, (3, -4), axis=(0, 1))
    signatures = (make_signature(reading_left, left_cells),
                  make_signature(reading_right, right_cells))
    direct = plane_graph.match_plane_pair(signatures[0], signatures[1], quadrant=0,
                                          max_shift_cells=8)
    assert direct is not None and direct.cell_shift_rc == (-3, 4)
    graph = plane_graph.match_plane_graph(
        (reading_left, reading_right), signatures, max_shift_cells=8)
    q0 = next(match for match in graph if match.quadrant_of_right == 0)
    assert q0.cell_shift_rc == direct.cell_shift_rc

    with pytest.raises(ValueError, match=r"integer \(row, column\) pair"):
        replace(direct, cell_shift_rc=(True, 4))


def test_match_graph_is_deterministic_and_typed():
    reading_left = make_reading("a", 0, -1.0, [(1.0, 6.0, 1.0)], share=0.6)
    reading_right = make_reading("b", 0, -1.0, [(1.0, 6.0, 1.0)], share=0.6)
    reading_other = make_reading("c", 1, 1.0, [(1.0, 4.0, 1.0)], share=0.6)
    rng = np.random.default_rng(31)
    signatures = [
        make_signature(reading_left, rng.random((20, 20)), b=1.0),
        make_signature(reading_right, np.roll(rng.random((20, 20)), (1, 2), axis=(0, 1)), b=1.0),
        make_signature(reading_other, rng.random((20, 20)), b=1.0),
    ]
    readings = (reading_left, reading_right, reading_other)
    first = plane_graph.match_plane_graph(readings, signatures)
    second = plane_graph.match_plane_graph(readings, signatures)
    assert [(m.left, m.right, m.left_b, m.right_b, m.quadrant_of_right) for m in first] == \
        [(m.left, m.right, m.left_b, m.right_b, m.quadrant_of_right) for m in second]
    refusals: list = []
    plane_graph.match_plane_graph(readings, signatures, refusals=refusals)
    assert all(refusal.reason in plane_graph.MATCH_REFUSAL_REASONS for refusal in refusals)


@pytest.mark.parametrize("right_axis", range(3))
@pytest.mark.parametrize("quadrant", range(4))
@pytest.mark.parametrize("left_side", (-1.0, 1.0))
@pytest.mark.parametrize("right_side", (-1.0, 1.0))
def test_same_axis_delta_uses_right_orientation_sign(
        right_axis, quadrant, left_side, right_side):
    left_axis, sigma = plane_graph.axis_transform(right_axis, quadrant)
    left = make_reading("a", left_axis, left_side, [(1.0, 2.0, 1.0)])
    right = make_reading("b", right_axis, right_side, [(1.0, 3.0, 1.0)])
    readings = {("a", 0): left, ("b", 0): right}
    match = make_match("a", "b", left_axis, right_axis, quadrant, 0, 0)

    assert plane_graph._delta(match, readings) == pytest.approx(
        left_side * 2.0 - sigma * right_side * 3.0)

    incompatible_readings = dict(readings)
    incompatible_readings[("a", 0)] = make_reading(
        "a", (left_axis + 1) % 3, left_side, [(1.0, 2.0, 1.0)])
    assert plane_graph._delta(match, incompatible_readings) is None


def test_same_axis_filter_keeps_mutually_exclusive_astra_counterexample():
    # World planes 0/20 observed from cameras 5/6.  Relative quadrants 0 and 2
    # are mutually exclusive orientation hypotheses, not competing measurements.
    readings = {
        ("a", 0): make_reading("a", 0, -1.0, [(1.0, 5.0, 1.0)]),
        ("a", 1): make_reading("a", 0, 1.0, [(1.0, 15.0, 1.0)]),
        ("b", 0): make_reading("b", 0, -1.0, [(1.0, 6.0, 1.0)]),
        ("b", 1): make_reading("b", 0, 1.0, [(1.0, 14.0, 1.0)]),
    }
    matches = [
        make_match("a", "b", 0, 0, 0, 0, 0, score=0.99),
        make_match("a", "b", 0, 0, 2, 1, 1, score=0.91),
    ]

    for candidate_order in (matches, matches[::-1]):
        refusals = []
        kept = plane_graph._apply_same_axis_filter(
            list(candidate_order), readings, refusals)
        assert {match.quadrant_of_right for match in kept} == {0, 2}
        assert refusals == []


def test_same_axis_filter_drops_real_disagreement_and_tie_deterministically():
    readings = {
        ("a", 0): make_reading("a", 0, -1.0, [(1.0, 5.0, 1.0)]),
        ("a", 1): make_reading("a", 0, 1.0, [(1.0, 15.0, 1.0)]),
        ("b", 0): make_reading("b", 0, -1.0, [(1.0, 6.0, 1.0)]),
        ("b", 1): make_reading("b", 0, 1.0, [(1.0, 10.0, 1.0)]),
    }
    higher = make_match("a", "b", 0, 0, 0, 0, 0, score=0.99)
    lower = make_match("a", "b", 0, 0, 0, 1, 1, score=0.91)
    for candidate_order in ([higher, lower], [lower, higher]):
        refusals = []
        kept = plane_graph._apply_same_axis_filter(
            list(candidate_order), readings, refusals)
        assert kept == [higher]
        assert len(refusals) == 1
        assert refusals[0].reason == "same_axis_mismatch"

    tied = replace(lower, score=higher.score)
    for candidate_order in ([higher, tied], [tied, higher]):
        refusals = []
        assert plane_graph._apply_same_axis_filter(
            list(candidate_order), readings, refusals) == []
        assert len(refusals) == 2
        assert {refusal.reason for refusal in refusals} == {"same_axis_mismatch"}


def test_public_matcher_does_not_cross_delete_relative_quadrants(monkeypatch):
    left = make_reading("a", 0, -1.0, [(1.0, 5.0, 1.0)], share=0.6)
    right = make_reading("b", 0, -1.0, [(1.0, 6.0, 1.0)], share=0.6)
    cells = np.arange(400, dtype=float).reshape(20, 20)
    signatures = (make_signature(left, cells), make_signature(right, cells))

    def controlled_match(left_signature, right_signature, *, quadrant,
                         max_shift_cells, refusals):
        if quadrant not in (0, 2):
            return None
        return make_match(
            left_signature.reading.frame, right_signature.reading.frame,
            left_signature.reading.axis, right_signature.reading.axis,
            quadrant, 0, 0, score=0.99 if quadrant == 0 else 0.91)

    monkeypatch.setattr(plane_graph, "match_plane_pair", controlled_match)
    refusals = []
    matches = plane_graph.match_plane_graph(
        (left, right), signatures, refusals=refusals)

    assert tuple(match.quadrant_of_right for match in matches) == (0, 2)
    assert refusals == []


def test_resolved_class_margin_uses_the_retained_alternatives():
    readings, matches, _eyes = lattice_component()
    solution = plane_graph.solve_component(readings, matches)
    calibration = {frame: object() for frame in solution.frames}
    for frame in solution.frames:
        quadrant, margin = plane_graph.resolved_class(solution, frame, calibration)
        assert quadrant == solution.quadrants[frame]
        assert margin == float("inf") or margin >= 0.0
    assert plane_graph.resolved_class(solution, "missing", calibration) == (0, 0.0)
    assert plane_graph.resolved_class(solution, "v0", {}) == (0, 0.0)


def test_component_abstention_vocabulary_is_complete():
    assert set(plane_graph.COMPONENT_ABSTENTION_REASONS) == {
        "no_plane_reading", "no_matches", "insufficient_structural_share",
        "repetition_unresolved", "plane_identity_ambiguous", "same_axis_mismatch",
        "ambiguous_quadrant", "plane_integer_residual", "insufficient_axis_coverage",
        "b_unresolved", "component_too_large", "quadrant_constraint_inconsistent",
        "endpoint_period_conflict",
    }
    assert set(plane_graph.MATCH_REFUSAL_REASONS) == {
        "axis_mismatch", "invalid_distance_alternative", "insufficient_overlap",
        "no_signal", "repetition_unresolved", "ambiguity_check_inconclusive",
        "same_axis_mismatch",
    }
    assert plane_graph.solve_component([], []).reason == "no_plane_reading"


@pytest.mark.parametrize("left_samples,right_samples", ((5, 9), (17, 64), (128, 512)))
def test_cell_grid_homography_is_camera_independent_for_sample_variants(
        left_samples, right_samples):
    extent = 1.25
    left_period, right_period = 3.75, 4.5
    left_b, right_b = 0.5, 2.0
    left_distance = left_b / (
        left_period * lattice.rectified_pixel_step(extent, left_samples))
    right_distance = right_b / (
        right_period * lattice.rectified_pixel_step(extent, right_samples))
    left_reading = replace(
        make_reading("left", 2, 1.0, [(left_b, left_distance, 1.0)]),
        extent=extent, samples=left_samples, period_pixels=left_period)
    right_reading = replace(
        make_reading("right", 2, 1.0, [(right_b, right_distance, 1.0)]),
        extent=extent, samples=right_samples, period_pixels=right_period)
    rng = np.random.default_rng(88)
    left = replace(make_signature(left_reading, rng.random((24, 24)), b=left_b),
                   origin_uv=(2.25, -1.5))
    right = replace(make_signature(right_reading, rng.random((24, 24)), b=right_b),
                    origin_uv=(-0.75, 3.0))

    expected = np.asarray(((1.0, 0.0, right.origin_uv[0] - left.origin_uv[0]),
                           (0.0, 1.0, right.origin_uv[1] - left.origin_uv[1]),
                           (0.0, 0.0, 1.0)))
    actual = plane_graph.cell_grid_homography(left, right, 0)
    np.testing.assert_array_equal(actual, expected)
    actual_warp, actual_coverage = lattice_planes.warp_signature(
        left, actual, right.cells.shape)
    expected_warp, expected_coverage = lattice_planes.warp_signature(
        left, expected, right.cells.shape)
    np.testing.assert_array_equal(actual_warp, expected_warp)
    np.testing.assert_array_equal(actual_coverage, expected_coverage)


@pytest.mark.parametrize("left_axis,quadrant,expected", [
    (0, 0, ((1, 0), (0, 1))),
    (0, 1, ((0, 1), (1, 0))),
    (0, 2, ((1, 0), (0, -1))),
    (0, 3, ((0, 1), (-1, 0))),
    (1, 0, ((1, 0), (0, 1))),
    (1, 1, ((0, -1), (1, 0))),
    (1, 2, ((-1, 0), (0, -1))),
    (1, 3, ((0, 1), (-1, 0))),
    (2, 0, ((1, 0), (0, 1))),
    (2, 1, ((0, -1), (1, 0))),
    (2, 2, ((-1, 0), (0, 1))),
    (2, 3, ((0, 1), (1, 0))),
])
def test_block_grid_map_matches_literal_compatible_quadrant_table(
        left_axis, quadrant, expected):
    right_axis, _sign = plane_graph.axis_transform(left_axis, quadrant)
    left = make_signature(make_reading("l", left_axis, 1.0, [(1.0, 2.0, 1.0)]),
                          np.arange(30, dtype=np.float32).reshape(5, 6))
    right = make_signature(make_reading("r", right_axis, 1.0, [(1.0, 7.0, 1.0)]),
                           np.arange(42, dtype=np.float32).reshape(6, 7))
    left = replace(left, origin_uv=(1.25, -2.5))
    right = replace(right, origin_uv=(-0.75, 3.5))
    actual = plane_graph.cell_grid_homography(left, right, quadrant)
    expected_p = np.asarray(expected, dtype=np.float64)
    expected_t = expected_p @ np.asarray(right.origin_uv) - np.asarray(left.origin_uv)
    expected_h = np.eye(3)
    expected_h[:2, :2] = expected_p
    expected_h[:2, 2] = expected_t
    np.testing.assert_array_equal(actual, expected_h)
    for right_point in ((0.0, 0.0), (2.5, 1.25), (6.0, 5.0)):
        homogeneous = np.asarray((*right_point, 1.0))
        left_point = actual @ homogeneous
        np.testing.assert_allclose(left_point[:2],
            expected_p @ (np.asarray(right_point) + right.origin_uv) - left.origin_uv,
            rtol=0.0, atol=0.0)
        np.testing.assert_allclose(np.linalg.inv(actual) @ left_point, homogeneous,
                                   rtol=0.0, atol=1e-12)


def test_grid_homography_rejects_noncanonical_quadrants_axes_origins_and_grids():
    left = make_signature(make_reading("l", 2, 1.0, [(1.0, 2.0, 1.0)]),
                          np.arange(16, dtype=np.float32).reshape(4, 4))
    right = make_signature(make_reading("r", 2, 1.0, [(1.0, 4.0, 1.0)]),
                           np.arange(16, dtype=np.float32).reshape(4, 4))
    for q in (True, 1.0, -1, 4, None):
        with pytest.raises(ValueError):
            plane_graph.cell_grid_homography(left, right, q)
    with pytest.raises(ValueError, match="incompatible"):
        plane_graph.cell_grid_homography(left, replace(right, reading=replace(
            right.reading, axis=0)), 0)
    for origin in ((float("nan"), 0.0), (0.0,), [0.0, 0.0]):
        with pytest.raises(ValueError, match="origin"):
            plane_graph.cell_grid_homography(left, replace(right, origin_uv=origin), 0)
    with pytest.raises(ValueError, match="shape-matched"):
        plane_graph.cell_grid_homography(left, replace(right, interior=right.interior[:, :-1]), 0)


def test_grid_map_is_independent_of_measurement_geometry_while_c1_binds_it():
    from mc_photogrammetry.reconstruction import plane_axis_support

    reading = make_reading("left", 2, 1.0, [(1.0, 4.0, 1.0)], period=60.0)
    base = replace(make_signature(reading, np.arange(64, dtype=np.float32).reshape(8, 8)),
                   origin_uv=(0.25, -0.75))
    right = replace(make_signature(
        make_reading("right", 2, 1.0, [(1.0, 9.0, 1.0)]),
        np.arange(64, dtype=np.float32).reshape(8, 8)), origin_uv=(-1.5, 2.25))
    base_map = plane_graph.cell_grid_homography(base, right, 0)
    base_geometry = plane_axis_support._signature_geometry_digest(base, 0)
    assert base_geometry is not None
    rect = reading.rectification
    changed_readings = (
        replace(reading, rectification=replace(rect, focal=700.0)),
        replace(reading, rectification=replace(rect, principal_point=(17.0, 29.0))),
        replace(reading, rectification=replace(rect, centre=(5.0, 7.0))),
        replace(reading, rectification=replace(
            rect, axis_matrix=((0.0, -1.0, 0.0),
                               (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))),
        replace(reading, samples=256),
        replace(reading, period_pixels=77.0),
        replace(reading, extent=1.75),
        replace(reading, side=-1.0),
        replace(reading, alternatives=(
            lattice_planes.DistanceAlternative(1.0, 7.0, 1.0),)),
    )
    for changed_reading in changed_readings:
        changed = replace(base, reading=changed_reading)
        changed_map = plane_graph.cell_grid_homography(changed, right, 0)
        assert changed_map.tobytes() == base_map.tobytes()
        changed_geometry = plane_axis_support._signature_geometry_digest(changed, 0)
        assert changed_geometry is not None
        assert changed_geometry != base_geometry


def test_grid_mapping_helper_has_no_external_authority_or_io_dependencies():
    from mc_photogrammetry.reconstruction import cell_grid_homography

    assert cell_grid_homography is plane_graph.cell_grid_homography
    module = ast.parse(Path(plane_graph.__file__).read_text(encoding="utf-8"))
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(module)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert not imports & {
        "PIL", "cv2", "torch", "yaml", "jsonschema",
        "evaluation_frozen_planes", "frozen_image_runner",
    }
    helper = next(node for node in module.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "cell_grid_homography")
    assert len(helper.args.args) == 3 and helper.args.defaults == []
    called = {
        node.func.id if isinstance(node.func, ast.Name)
        else node.func.attr if isinstance(node.func, ast.Attribute) else ""
        for node in ast.walk(helper) if isinstance(node, ast.Call)
    }
    assert not called & {
        "imread", "imdecode", "open", "load_camera_manifest",
        "load_scene_transform", "load_and_verify_frozen_image",
        "reconstruct_from_calibrated_views", "evaluate_frozen_planes",
    }


def test_trailing_signatureless_readings_keep_matcher_and_solver_plane_indices():
    # Candidate signatures cover axes 0 and 1. The axis-2 observations have no
    # usable signature and sort after them; the graph match's numeric identity
    # must still resolve to the same reading at the solver boundary.
    readings = tuple(make_reading(frame, axis, 1.0, [(1.0, 4.0, 1.0)])
                     for frame in ("a", "b") for axis in range(3))
    by_frame_axis = {(reading.frame, reading.axis): reading for reading in readings}
    rng = np.random.default_rng(744)
    cells_by_axis = {axis: rng.normal(size=(32, 32)) for axis in (0, 1)}
    signatures = tuple(
        make_signature(by_frame_axis[(frame, axis)], cells_by_axis[axis])
        for frame in ("a", "b") for axis in (0, 1))
    matches = plane_graph.match_plane_graph(readings, signatures, max_shift_cells=4)
    assert matches
    ordered = {
        frame: tuple(sorted((reading for reading in readings if reading.frame == frame),
                            key=lattice_planes.reading_order_key))
        for frame in ("a", "b")
    }
    assert all(ordered[match.left][match.left_plane]
               is by_frame_axis[(match.left, match.left_axis)]
               and ordered[match.right][match.right_plane]
               is by_frame_axis[(match.right, match.right_axis)]
               for match in matches)
    assert sum(1 for reading in readings
               if not any(signature.reading is reading for signature in signatures)) == 2
    solved = plane_graph.solve_component(readings, matches)
    assert solved.reason != "same_axis_mismatch"


def test_leading_signatureless_reading_preserves_match_identity_at_solver_boundary():
    readings = tuple(make_reading(frame, axis, 1.0, [(1.0, 4.0, 1.0)])
                     for frame in ("a", "b") for axis in range(3))
    by_frame_axis = {(reading.frame, reading.axis): reading for reading in readings}
    rng = np.random.default_rng(745)
    cells_by_axis = {axis: rng.normal(size=(32, 32)) for axis in (1, 2)}
    # Axis 0 has no signature and sorts before both matched readings.
    signatures = tuple(
        make_signature(by_frame_axis[(frame, axis)], cells_by_axis[axis])
        for frame in ("a", "b") for axis in (1, 2))
    matches = plane_graph.match_plane_graph(readings, signatures, max_shift_cells=4)
    assert matches
    solver_ordered = {
        frame: tuple(sorted((reading for reading in readings
                             if reading.frame == frame),
                            key=lattice_planes.reading_order_key))
        for frame in ("a", "b")
    }
    # Both matcher and solver use all three readings, so leading signatureless
    # axis 0 leaves the matched axes at indices 1 and 2.
    for match in matches:
        assert match.left_plane == match.left_axis
        assert match.right_plane == match.right_axis
        assert solver_ordered[match.left][match.left_plane] \
            is by_frame_axis[(match.left, match.left_axis)]
        assert solver_ordered[match.right][match.right_plane] \
            is by_frame_axis[(match.right, match.right_axis)]
    solved = plane_graph.solve_component(readings, matches)
    assert solved.reason != "no_matches"


@pytest.mark.parametrize("reading_specs,signature_specs", (
    (((0, 1.0), (1, 1.0), (2, 1.0)), ((0, 1.0), (1, 1.0))),
    (((0, 1.0), (1, -1.0), (1, 1.0), (2, 1.0)),
     ((0, 1.0), (2, 1.0))),
    (((0, -1.0), (0, 1.0), (1, -1.0), (1, 1.0), (2, -1.0), (2, 1.0)),
     ((1, 1.0), (2, 1.0))),
))
def test_complete_reading_index_is_permutation_stable(
        reading_specs, signature_specs):
    readings = tuple(make_reading(frame, axis, side,
                                  [(1.0, 4.0 + axis + side, 1.0)])
                     for frame in ("a", "b") for axis, side in reading_specs)
    by_frame_identity = {(reading.frame, reading.axis, reading.side): reading
                         for reading in readings}
    rng = np.random.default_rng(902)
    cells = {identity: rng.normal(size=(24, 24)) for identity in signature_specs}
    signatures = tuple(
        make_signature(by_frame_identity[(frame, axis, side)], cells[(axis, side)])
        for frame in ("a", "b") for axis, side in signature_specs)
    expected = plane_graph.match_plane_graph(readings, signatures)
    permuted = plane_graph.match_plane_graph(readings[::-1], signatures[::-1])
    assert permuted == expected
    for match in expected:
        left_order = tuple(sorted((item for item in readings if item.frame == match.left),
                                  key=lattice_planes.reading_order_key))
        right_order = tuple(sorted((item for item in readings if item.frame == match.right),
                                   key=lattice_planes.reading_order_key))
        assert left_order[match.left_plane] is by_frame_identity[
            (match.left, match.left_axis, 1.0)]
        assert right_order[match.right_plane] is by_frame_identity[
            (match.right, match.right_axis, 1.0)]


def test_complete_universe_rejects_foreign_duplicate_and_ambiguous_readings():
    left = make_reading("a", 1, 1.0, [(1.0, 4.0, 1.0)])
    right = make_reading("b", 1, 1.0, [(1.0, 4.0, 1.0)])
    cells = np.random.default_rng(903).normal(size=(24, 24))
    left_signature, right_signature = (make_signature(left, cells),
                                       make_signature(right, cells))
    with pytest.raises(TypeError):
        plane_graph.match_plane_graph((left_signature, right_signature))
    with pytest.raises(ValueError, match="foreign"):
        plane_graph.match_plane_graph((left,), (right_signature,))
    with pytest.raises(ValueError, match="repeats"):
        plane_graph.match_plane_graph((left, left, right),
                                      (left_signature, right_signature))
    with pytest.raises(ValueError, match="ambiguous canonical order key"):
        plane_graph.match_plane_graph((left, replace(left), right),
                                      (left_signature, right_signature))
    with pytest.raises(ValueError, match="non-empty string frame"):
        plane_graph.match_plane_graph((replace(left, frame=""), right),
                                      (left_signature, right_signature))
    with pytest.raises(ValueError, match="finite, orderable"):
        plane_graph.match_plane_graph(
            (replace(left, confidence=float("nan")), right),
            (left_signature, right_signature))
    with pytest.raises(ValueError, match="signature b"):
        plane_graph.match_plane_graph(
            (left, right),
            (replace(left_signature, blocks_per_period=True), right_signature))


def test_complete_universe_validation_finishes_before_correlation(monkeypatch):
    left = make_reading("a", 1, 1.0, [(1.0, 4.0, 1.0)])
    right = make_reading("b", 1, 1.0, [(1.0, 4.0, 1.0)])
    signatures = (make_signature(left, np.ones((12, 12))),
                  make_signature(right, np.ones((12, 12))))
    malformed_late_reading = replace(make_reading(
        "b", 2, 1.0, [(1.0, 4.0, 1.0)]), confidence=float("inf"))

    def correlation_must_not_start(*_args, **_kwargs):
        raise AssertionError("correlation started before full-universe validation")

    monkeypatch.setattr(plane_graph, "match_plane_pair", correlation_must_not_start)
    with pytest.raises(ValueError, match="finite, orderable"):
        plane_graph.match_plane_graph(
            (left, right, malformed_late_reading), signatures)


def test_signature_alternatives_share_one_reading_index_and_bad_b_is_only_refused():
    left = make_reading("a", 1, 1.0, [(1.0, 4.0, 1.0), (2.0, 8.0, 0.8)])
    right = make_reading("b", 1, 1.0, [(1.0, 4.0, 1.0), (2.0, 8.0, 0.8)])
    cells = np.random.default_rng(904).random((24, 24))
    readings = (left, right)
    signatures = tuple(make_signature(reading, cells, b=b)
                       for reading in readings for b in (1.0, 3.0))
    refusals = []
    matches = plane_graph.match_plane_graph(readings, signatures, refusals=refusals)
    assert matches
    assert {match.left_plane for match in matches} == {0}
    assert {match.right_plane for match in matches} == {0}
    assert {match.left_b for match in matches} == {1.0}
    assert all(plane_graph._canonical_reading_index(readings)[0][
        (match.left, match.left_plane)] is left for match in matches)
    assert any(refusal.reason == "invalid_distance_alternative"
               for refusal in refusals)
    match = next(value for value in matches if value.quadrant_of_right == 0)
    for field, invalid in (("left_b", True), ("right_b", True),
                           ("left_b", float("nan")),
                           ("right_b", float("nan"))):
        with pytest.raises(ValueError, match="selected b"):
            plane_graph.solve_component(
                readings, (replace(match, **{field: invalid}),))


@pytest.mark.parametrize("alternatives", (
    ((0.5, 3.0, 1.0),),
    ((1.0, 3.0, 1.0), (1.0, 4.0, 0.9)),
))
def test_grid_map_ignores_distance_alternatives_but_match_still_refuses_them(alternatives):
    left_reading = make_reading("left", 0, 1.0, alternatives)
    right_reading = make_reading("right", 0, 1.0, [(1.0, 2.0, 1.0)])
    left = make_signature(left_reading, np.ones((12, 12)), b=1.0)
    right = make_signature(right_reading, np.ones((12, 12)), b=1.0)
    homography = plane_graph.cell_grid_homography(left, right, 0)
    np.testing.assert_array_equal(homography, np.eye(3))
    refusals = []
    assert plane_graph.match_plane_pair(left, right, quadrant=0, refusals=refusals) is None
    assert len(refusals) == 1
    assert refusals[0].reason == "invalid_distance_alternative"


def test_s6_class_constrained_bestof_selects_the_true_minority_and_solves():
    left_model = CameraModel(position=(0.0, 0.0, 0.0), yaw_degrees=0.0, pitch_degrees=0.0,
                             fov_degrees=70.0)
    right_model = CameraModel(position=(3.0, 0.0, 1.0), yaw_degrees=90.0, pitch_degrees=0.0,
                              fov_degrees=70.0)
    # The estimator reports yaw modulo 90, so the right view's *calibrated*
    # rotation is R(0) and the quadrant hypothesis k = 1 supplies the missing
    # R(0) @ Ry(90) = R(90) that matches the true camera.
    right_calibrated = CameraModel(position=(3.0, 0.0, 1.0), yaw_degrees=0.0,
                                   pitch_degrees=0.0, fov_degrees=70.0)
    rng = np.random.default_rng(29)
    points = rng.uniform(-6.0, 6.0, (30, 3)) + np.asarray((0.0, 0.0, 20.0))
    left_pixels = project_points([tuple(point) for point in points], left_model,
                                 width=WIDTH, height=HEIGHT)
    right_pixels = project_points([tuple(point) for point in points], right_model,
                                  width=WIDTH, height=HEIGHT)
    paired = [(left[:2], right[:2]) for left, right in zip(left_pixels, right_pixels)
              if left is not None and right is not None]
    paired = paired[:20]
    left_points = np.asarray([left for left, _right in paired], dtype=float)
    right_points = np.asarray([right for _left, right in paired], dtype=float)
    count = len(paired)

    left_intrinsics = intrinsics_for(left_model, WIDTH, HEIGHT)
    right_intrinsics = intrinsics_for(right_model, WIDTH, HEIGHT)
    left_rays = _rays(left_points, left_intrinsics)
    right_rays = _rays(right_points, right_intrinsics)

    edges = []
    distances = []
    for index in range(count):
        edges.append((index, index))
        distances.append(0.1)
    for index in range(count):
        if index % 5 < 2:  # 40 % impostor edges
            impostor = (index + 7) % count
            if impostor != index:
                edges.append((index, impostor))
                distances.append(0.9)
    neighbors = _Neighbors(np.asarray(edges, dtype=np.int64), np.asarray(distances, dtype=float))
    features = plane_graph.ClassConstrainedFeatures(
        left_keypoints=left_points, right_keypoints=right_points,
        left_rays=left_rays, right_rays=right_rays,
        rotation_left=camera_rotation_matrix(left_model),
        rotation_right=camera_rotation_matrix(right_calibrated))

    selected = plane_graph.class_constrained_bestof(features, neighbors, 1, residual_degrees=1.0)
    assert len(selected) >= 8
    left_lookup = {tuple(left_points[index]): index for index in range(count)}
    right_lookup = {tuple(right_points[index]): index for index in range(count)}
    left_indices = [left_lookup[(float(row[0]), float(row[1]))] for row in selected]
    right_indices = [right_lookup[(float(row[2]), float(row[3]))] for row in selected]
    assert len(set(left_indices)) == len(selected)
    assert len(set(right_indices)) == len(selected)
    assert left_indices == right_indices

    left_estimate = CameraEstimate(
        intrinsics=CameraIntrinsics((WIDTH, HEIGHT), (left_intrinsics.focal_length_pixels),
                                    left_intrinsics.principal_point_pixels),
        pose=CameraPose(position=None, yaw_degrees=0.0, pitch_degrees=0.0, roll_degrees=0.0))
    right_estimate = CameraEstimate(
        intrinsics=CameraIntrinsics((WIDTH, HEIGHT), (right_intrinsics.focal_length_pixels),
                                    right_intrinsics.principal_point_pixels),
        pose=CameraPose(position=None, yaw_degrees=0.0, pitch_degrees=0.0, roll_degrees=0.0))
    pose = solve_pair(left_estimate, right_estimate, selected)
    assert pose.solved is True
    assert pose.right_yaw_class == 1
    assert pose.inlier_fraction >= 0.5


def test_proposal_pipeline_either_solves_or_abstains_with_a_typed_reason():
    """The image-level route is honest: a solved component or a typed refusal."""
    readings_by_frame = {}
    signatures = []
    positions = {"v0": ((3.6, 0.7, 3.6), 10.0, -25.0),
                 "v1": ((4.5, 0.9, 4.2), 14.0, -22.0),
                 "v2": ((5.5, 0.6, 3.0), 6.0, -28.0)}
    for frame, (position, yaw, pitch) in positions.items():
        camera = CameraModel(position=position, yaw_degrees=yaw, pitch_degrees=pitch,
                             fov_degrees=70.0)
        image = _render_room(camera)
        config = _calibration(camera)
        found = lattice_planes.frame_plane_readings(image, _usable(), config, frame=frame)
        readings_by_frame[frame] = found
        for reading in found:
            for alternative in reading.alternatives:
                signatures.append(lattice_planes.plane_signature(
                    image, reading, alternative.blocks_per_period, usable=_usable()))
    readings = [reading for frame in sorted(readings_by_frame)
                for reading in readings_by_frame[frame]]
    refusals: list = []
    matches = plane_graph.match_plane_graph(readings, signatures, refusals=refusals)
    solution = plane_graph.solve_component(readings, matches)
    if solution.reason == "":
        assert solution.scale_known is True
        assert all(value > 0.9 for value in solution.axis_resultants.values())
    else:
        assert solution.reason in plane_graph.COMPONENT_ABSTENTION_REASONS
    assert all(refusal.reason in plane_graph.MATCH_REFUSAL_REASONS for refusal in refusals)


def test_region_growth_proposals_bridge_through_readings_signatures_and_graph():
    """Exercise C4 masks through the additive image-only DS-29 consumer."""
    readings = []
    signatures = []
    positions = {
        "v0": ((3.6, 0.7, 3.6), 10.0, -25.0),
        "v1": ((4.5, 0.9, 4.2), 14.0, -22.0),
    }
    for frame, (position, yaw, pitch) in positions.items():
        camera = CameraModel(position=position, yaw_degrees=yaw, pitch_degrees=pitch,
                             fov_degrees=70.0)
        # The synthetic renderer returns luma; triplicate channels to exercise
        # the proposal API's explicit RGB contract without adding information.
        image = np.repeat(_render_room(camera)[..., None], 3, axis=2)
        usable = _usable()
        calibration = _calibration(camera)
        proposals = propose_image_planes_v2(
            image, usable, calibration, max_candidates=8)
        assert proposals.status == "ok" and proposals.proposals
        frame_readings = lattice_planes.frame_plane_readings_from_proposals(
            image, usable, calibration, proposals, frame=frame)
        assert frame_readings
        readings.extend(frame_readings)
        for reading in frame_readings:
            alternative = reading.alternatives[0]
            signatures.append(lattice_planes.plane_signature(
                image, reading, alternative.blocks_per_period, usable=usable))

    refusals = []
    matches = plane_graph.match_plane_graph(readings, signatures, refusals=refusals)
    assert matches
    solution = plane_graph.solve_component(readings, matches)
    if solution.reason == "":
        assert solution.scale_known is True
    else:
        assert solution.reason in plane_graph.COMPONENT_ABSTENTION_REASONS
    assert all(refusal.reason in plane_graph.MATCH_REFUSAL_REASONS for refusal in refusals)


class _Neighbors:
    def __init__(self, edges, distances):
        self.edges = edges
        self.distances = distances


def _rays(points, intrinsics):
    focal_x, focal_y = intrinsics.focal_length_pixels
    centre_x, centre_y = intrinsics.principal_point_pixels
    x = (points[:, 0] - centre_x) / focal_x
    y = -(points[:, 1] - centre_y) / focal_y
    rays = np.column_stack((x, y, np.ones(len(points))))
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


_ROOM = (0.0, 12.0, 0.0, 6.0, 0.0, 12.0)
_USABLE: dict[str, np.ndarray] = {}


def _usable() -> np.ndarray:
    if "mask" not in _USABLE:
        from mc_photogrammetry.vision.gui_mask import create_gui_mask

        _USABLE["mask"] = ~create_gui_mask(HEIGHT, WIDTH)
    return _USABLE["mask"]


def _calibration(camera: CameraModel) -> CalibrationResult:
    intrinsics = intrinsics_for(camera, WIDTH, HEIGHT)
    return CalibrationResult(
        focal_pixels=float(intrinsics.focal_length_pixels[0]),
        yaw_degrees=float(camera.yaw_degrees % 90.0), pitch_degrees=float(camera.pitch_degrees),
        width=WIDTH, height=HEIGHT, inliers=100, segments=200,
        median_residual=0.5, roll_degrees=0.0, principal_point=None)


def _texture(material, size=16):
    val = np.full((size, size), 150.0 + 6.0 * (material % 4))
    shift = (material * 5) % 5
    val[4 + (shift % 2), :] = 128.0 + 6.0 * (material % 3)
    val[:, 11 - (shift % 2)] = 134.0 + 4.0 * (material % 2)
    val[0, :] = 25.0
    val[:, 0] = 25.0
    return val.astype(np.uint8)


def _render_room(camera: CameraModel):
    x0, x1, y0, y1, z0, z1 = _ROOM
    rotation = camera_rotation_matrix(camera)
    intrinsics = intrinsics_for(camera, WIDTH, HEIGHT)
    fx, fy = intrinsics.focal_length_pixels
    cx, cy = intrinsics.principal_point_pixels
    xs, ys = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    world = np.stack([(xs - cx) / fx, -(ys - cy) / fy, np.ones_like(xs, dtype=float)],
                     axis=-1) @ rotation
    eye = np.asarray(camera.position, float) + np.asarray([0.0, camera.eye_height, 0.0])
    best = np.full((HEIGHT, WIDTH), np.inf)
    color = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for axis, value in [(0, x0), (0, x1), (1, y0), (1, y1), (2, z0), (2, z1)]:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (value - eye[axis]) / world[..., axis]
        hit = np.isfinite(t) & (t > 1e-6) & (t < best)
        if not hit.any():
            continue
        point = eye[None, None, :] + np.where(hit, t, 0.0)[..., None] * world
        inside = np.ones((HEIGHT, WIDTH), bool)
        bounds = {0: (x0, x1), 1: (y0, y1), 2: (z0, z1)}
        for other in range(3):
            if other == axis:
                continue
            lo, hi = bounds[other]
            inside &= (point[..., other] >= lo - 1e-6) & (point[..., other] <= hi + 1e-6)
        use = hit & inside
        if not use.any():
            continue
        others = [a for a in range(3) if a != axis]
        u, v = point[..., others[0]], point[..., others[1]]
        bx, by = np.floor(u).astype(int), np.floor(v).astype(int)
        chosen = (bx * 3 + by * 5) % 3
        tu = np.clip((np.mod(u, 1.0) * 16).astype(int), 0, 15)
        tv = np.clip((np.mod(v, 1.0) * 16).astype(int), 0, 15)
        for material in np.unique(chosen):
            tex = _texture(int(material))
            pick = use & (chosen == material)
            if pick.any():
                color[pick] = tex[tv[pick], tu[pick]]
        best[use] = t[use]
    return color
