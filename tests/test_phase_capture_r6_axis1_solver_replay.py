"""DS-IMG-GEOM-PHASE-CAPTURE-R6 evaluation-only synthetic axis-1 solver replay.

Isolated counterfactual control for the three captured axis-1 edges of the R4
observer run. It drives the frozen public solver
:func:`plane_graph.solve_component` from a synthetic active-edge subset built out
of the four captured axis-1 readings

* ``A = image-0000`` plane ``p2`` (axis 1, side ``-1``),
* ``B = image-0001`` plane ``p3`` (axis 1, side ``+1``),
* ``C = image-0002`` plane ``p2`` (axis 1, side ``-1``),
* ``D = image-0002`` plane ``p3`` (axis 1, side ``+1``),

and the three captured edges

* match 5:  A(``b``=0.25) -- B(``b``=0.125), relative quadrant 3,
* match 19: A(``b``=0.5)  -- C(``b``=0.25),  relative quadrant 1,
* match 33: B(``b``=3.0)  -- D(``b``=0.125), relative quadrant 2,

pinned to the captured best-assignment quadrants ``[0, 3, 1]`` through the public
``quadrant_constraints`` argument. ``A`` and ``B`` carry their captured competing
labels (the artifact's endpoint alias conflict); ``C`` and ``D`` carry their
captured label plus an explicit factor-2 counterfactual whose distance is
rescaled as ``b * captured_distance / captured_b`` so the measured scale is held
fixed.

Evidence provenance (never read by this module; the captured numbers are pinned
constants): ``.tmp-geom-phase-capture-20261004/observation.json``,
``observation.json`` sha256 ``B345D4D7925C07CAA89706D1210D6F7F3ACA2B764CDA9BEE
82A393BE5CD27D69``, phase-capture ``record_sha256``
``5cbe0943a225abe565ad9f49bf7b0e136f305a992a52d48e51bdb3ed867b6535``. The R4
artifact reports ``trace.candidate_counts`` pre_admission 9592, post_admission
1928, post_same_axis 38, and the solver's returned axis-1 metrics resultant
``0.013942164981374504`` / residual ``0.24710226548952122`` with 6 face
observations over the 3 active axis-1 edges.

Scope and honest limitations. This is a synthetic active-edge *subset*, not a
real-image replay: only the 3 captured axis-1 edges exist here, so the reduced
component has no axis-0 or axis-2 evidence and cannot produce the real run's
``plane_integer_residual`` reason. The frozen solver therefore returns the typed
subset reason ``insufficient_axis_coverage`` while its axis-1 diagnostics still
reproduce the artifact exactly; the per-axis gate is evaluated separately. The
artifact omits the remaining 28 solver-input matches' ``b`` assignments and every
per-alternative support, so the full 38-match universe is not replayable from it.
No truth, camera, F3, direction, bounds, mask or evaluator metadata is read, no
image or reconstruction workload runs, and no production source, threshold,
matcher, solver or gate is changed. These controls are evaluation evidence only,
not production acceptance.
"""
from __future__ import annotations

import pytest

from mc_photogrammetry.reconstruction import lattice_planes, plane_graph

# --- Captured constants (pinned from the R4 artifact; no file I/O) -----------------

A_FRAME, A_SIDE = "image-0000", -1.0
B_FRAME, B_SIDE = "image-0001", 1.0
C_FRAME, C_SIDE = "image-0002", -1.0
D_FRAME, D_SIDE = "image-0002", 1.0

A_CAPTURED_B, A_CAPTURED_DISTANCE = 0.25, 0.7686558327326101
B_CAPTURED_B, B_CAPTURED_DISTANCE = 0.125, 0.1902686465519023
C_CAPTURED_B, C_CAPTURED_DISTANCE = 0.25, 0.3707098643189054
D_CAPTURED_B, D_CAPTURED_DISTANCE = 0.125, 0.2915112461703297

A_ALIAS_B, A_ALIAS_DISTANCE = 0.5, 1.5373116654652201
B_ALIAS_B, B_ALIAS_DISTANCE = 3.0, 4.5664475172456545

PINNED_QUADRANTS = {A_FRAME: 0, B_FRAME: 3, C_FRAME: 1}
QUADRANT_CONSTRAINTS = (
    (A_FRAME, B_FRAME, 3),
    (A_FRAME, C_FRAME, 1),
    (B_FRAME, C_FRAME, 2),
)

# The frozen solver's own defaults, restated so the gate under test is explicit.
MIN_RESULTANT = 0.90
INTEGER_TOLERANCE = 0.15

CAPTURED_AXIS1_RESULTANT = 0.013942164981374504
CAPTURED_AXIS1_RESIDUAL = 0.24710226548952122
CAPTURED_AXIS1_OBSERVATIONS = 6
CAPTURED_AXIS1_TRANSLATIONS = {
    A_FRAME: 0.0,
    B_FRAME: -2.453129010263554,
    C_FRAME: 0.32760272983272826,
}

# The artifact reports its metrics at ``compatibility_decimals = 6`` (1e-6); the
# reduced subset drives the identical three-edge least squares, so 1e-9 absolute
# is ~1000x tighter than that documented scale while absorbing BLAS round-off.
NUMERIC_REL_TOLERANCE = 1e-9
NUMERIC_ABS_TOLERANCE = 1e-12

_RECTIFICATION = lattice_planes.PlaneRectification(
    axis_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    focal=500.0, principal_point=(120.0, 90.0), centre=(0.0, 0.0))


def _rescaled_distance(captured_b, captured_distance, target_b):
    """Rescale a captured reading to ``target_b`` holding its measured scale."""
    return float(target_b) * (float(captured_distance) / float(captured_b))


C_FACTOR2_B = 0.5
C_FACTOR2_DISTANCE = _rescaled_distance(C_CAPTURED_B, C_CAPTURED_DISTANCE, C_FACTOR2_B)
D_FACTOR2_B = 0.25
D_FACTOR2_DISTANCE = _rescaled_distance(D_CAPTURED_B, D_CAPTURED_DISTANCE, D_FACTOR2_B)


def _reading(frame, side, alternatives):
    ordered = tuple(sorted(
        (lattice_planes.DistanceAlternative(float(b), float(d), float(s))
         for b, d, s in alternatives),
        key=lambda alternative: -alternative.blocks_per_period))
    return lattice_planes.PlaneReading(
        frame=frame, axis=1, side=float(side), alternatives=ordered,
        period_pixels=60.0, phase_pixels=(0.0, 0.0), extent=1.0, samples=512,
        polygon=None, coverage_cells=8, structural_share=0.6, confidence=0.9,
        rectification=_RECTIFICATION)


def _readings():
    """The four captured readings, ordered A, B, C, D."""
    return (
        _reading(A_FRAME, A_SIDE, ((A_CAPTURED_B, A_CAPTURED_DISTANCE, 1.0),
                                   (A_ALIAS_B, A_ALIAS_DISTANCE, 1.0))),
        _reading(B_FRAME, B_SIDE, ((B_CAPTURED_B, B_CAPTURED_DISTANCE, 1.0),
                                   (B_ALIAS_B, B_ALIAS_DISTANCE, 1.0))),
        _reading(C_FRAME, C_SIDE, ((C_CAPTURED_B, C_CAPTURED_DISTANCE, 1.0),
                                   (C_FACTOR2_B, C_FACTOR2_DISTANCE, 1.0))),
        _reading(D_FRAME, D_SIDE, ((D_CAPTURED_B, D_CAPTURED_DISTANCE, 1.0),
                                   (D_FACTOR2_B, D_FACTOR2_DISTANCE, 1.0))),
    )


def _plane_index(readings, target):
    """Resolve the ``(frame, index)`` address with the solver's own order key.

    The captured universe indexed A at ``p2``, B at ``p3`` and C/D at ``p2``/``p3``;
    this reduced four-reading subset renumbers each frame from zero.
    """
    frame_readings = sorted(
        (reading for reading in readings if reading.frame == target.frame),
        key=lattice_planes.reading_order_key)
    for index, reading in enumerate(frame_readings):
        if reading is target:
            return index
    raise AssertionError("target reading is not present in its frame")


def _match(left, right, left_plane, right_plane, quadrant, left_b, right_b):
    return plane_graph.PlaneMatch(
        left=left, right=right, left_axis=1, right_axis=1,
        left_b=float(left_b), right_b=float(right_b), quadrant_of_right=quadrant,
        psr=5.0, peak=1.0, second_peak=0.1, overlap_cells=64,
        integer_residual=None, score=1.0, left_plane=left_plane,
        right_plane=right_plane)


CAPTURED_SELECTION = {
    "A5": A_CAPTURED_B, "B5": B_CAPTURED_B,
    "A19": A_ALIAS_B, "C19": C_CAPTURED_B,
    "B33": B_ALIAS_B, "D33": D_CAPTURED_B,
}


def _selection(**overrides):
    selection = dict(CAPTURED_SELECTION)
    selection.update(overrides)
    return selection


def _solve(selection):
    readings = _readings()
    A, B, C, D = readings
    matches = (
        _match(A_FRAME, B_FRAME, _plane_index(readings, A), _plane_index(readings, B),
               3, selection["A5"], selection["B5"]),
        _match(A_FRAME, C_FRAME, _plane_index(readings, A), _plane_index(readings, C),
               1, selection["A19"], selection["C19"]),
        _match(B_FRAME, D_FRAME, _plane_index(readings, B), _plane_index(readings, D),
               2, selection["B33"], selection["D33"]),
    )
    return plane_graph.solve_component(
        list(readings), matches, integer_tolerance=INTEGER_TOLERANCE,
        min_resultant=MIN_RESULTANT, quadrant_constraints=QUADRANT_CONSTRAINTS)


def _edge_delta(left, left_b, right, right_b):
    """The solver's own ``s_left*d_left - s_right*d_right`` for one edge."""
    left_sign = plane_graph.axis_transform(left.axis, PINNED_QUADRANTS[left.frame])[1]
    right_sign = plane_graph.axis_transform(right.axis, PINNED_QUADRANTS[right.frame])[1]
    left_distance = left.distance_for(left_b)
    right_distance = right.distance_for(right_b)
    assert left_distance is not None and right_distance is not None
    return (left_sign * left.side * left_distance
            - right_sign * right.side * right_distance)


def _cycle_closure(selection):
    """Deduplicated triangle closure and its distance-to-integer defect."""
    A, B, C, D = _readings()
    closure = (_edge_delta(A, selection["A5"], B, selection["B5"])
               + _edge_delta(B, selection["B33"], D, selection["D33"])
               - _edge_delta(A, selection["A19"], C, selection["C19"]))
    return closure, abs(((closure + 0.5) % 1.0) - 0.5)


def _assert_axis1_gate_failed(solution):
    assert solution.axis_resultants[1] < MIN_RESULTANT
    assert solution.axis_residuals[1] > INTEGER_TOLERANCE


# --- Controls ----------------------------------------------------------------------


def test_captured_readings_resolve_in_the_reduced_subset():
    readings = _readings()
    A, B, C, D = readings
    assert [reading.axis for reading in readings] == [1, 1, 1, 1]
    # Reduced-subset renumbering: captured p2/p3 collapse to per-frame indices 0/0/0/1.
    assert _plane_index(readings, A) == 0
    assert _plane_index(readings, B) == 0
    assert _plane_index(readings, C) == 0
    assert _plane_index(readings, D) == 1
    assert A.distance_for(A_CAPTURED_B) == A_CAPTURED_DISTANCE
    assert A.distance_for(A_ALIAS_B) == A_ALIAS_DISTANCE
    assert B.distance_for(B_CAPTURED_B) == B_CAPTURED_DISTANCE
    assert B.distance_for(B_ALIAS_B) == B_ALIAS_DISTANCE
    assert C.distance_for(C_CAPTURED_B) == C_CAPTURED_DISTANCE
    assert C.distance_for(C_FACTOR2_B) == C_FACTOR2_DISTANCE
    assert D.distance_for(D_CAPTURED_B) == D_CAPTURED_DISTANCE
    assert D.distance_for(D_FACTOR2_B) == D_FACTOR2_DISTANCE
    # The A and B alternatives are the captured alias conflict; C/D are factor-2 arms.
    assert {alternative.blocks_per_period for alternative in A.alternatives} == {0.25, 0.5}
    assert {alternative.blocks_per_period for alternative in B.alternatives} == {0.125, 3.0}
    assert C_FACTOR2_DISTANCE == pytest.approx(2.0 * C_CAPTURED_DISTANCE, rel=0.0, abs=1e-15)
    assert D_FACTOR2_DISTANCE == pytest.approx(2.0 * D_CAPTURED_DISTANCE, rel=0.0, abs=1e-15)


def test_captured_axis1_edges_reproduce_artifact_metrics_with_typed_subset_abstention():
    solution = _solve(CAPTURED_SELECTION)
    assert dict(solution.quadrants) == PINNED_QUADRANTS
    # Only axis 1 exists in the reduced subset, so axes 0 and 2 are absent.
    assert set(solution.axis_resultants) == {1}
    assert set(solution.axis_residuals) == {1}
    assert solution.reason in plane_graph.COMPONENT_ABSTENTION_REASONS
    # Honest limitation: the reduced one-axis subset cannot reach the real run's
    # ``plane_integer_residual``; it stops at the earlier coverage predicate.
    assert solution.reason == "insufficient_axis_coverage"
    assert solution.axis_resultants[1] == pytest.approx(
        CAPTURED_AXIS1_RESULTANT, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
    assert solution.axis_residuals[1] == pytest.approx(
        CAPTURED_AXIS1_RESIDUAL, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
    assert solution.alternatives[0].axis_observations == (0, CAPTURED_AXIS1_OBSERVATIONS, 0)
    for frame, expected in CAPTURED_AXIS1_TRANSLATIONS.items():
        assert solution.translations[frame][1] == pytest.approx(
            expected, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
    _assert_axis1_gate_failed(solution)


def test_consistent_ab_labels_alone_leave_the_axis1_gate_failed():
    # A keeps b=0.25 on both edges and B keeps b=0.125 on both edges, removing the
    # captured endpoint alias conflict while holding every measured scale fixed.
    solution = _solve(_selection(A19=A_CAPTURED_B, B33=B_CAPTURED_B))
    assert solution.reason == "insufficient_axis_coverage"
    assert solution.axis_resultants[1] == pytest.approx(
        0.350066829204237, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
    assert solution.axis_residuals[1] == pytest.approx(
        0.18395030846923202, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
    # It improves the captured resultant but still fails the frozen gate.
    assert solution.axis_resultants[1] > CAPTURED_AXIS1_RESULTANT
    _assert_axis1_gate_failed(solution)


def test_factor2_c_or_d_labels_improve_resultant_but_never_pass_the_gate():
    factor2_c = _solve(_selection(C19=C_FACTOR2_B))
    assert factor2_c.reason == "insufficient_axis_coverage"
    assert factor2_c.axis_resultants[1] == pytest.approx(
        0.17439485288115192, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
    assert factor2_c.axis_residuals[1] == pytest.approx(
        0.22843909603434556, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
    assert factor2_c.axis_resultants[1] > CAPTURED_AXIS1_RESULTANT
    _assert_axis1_gate_failed(factor2_c)

    factor2_d = _solve(_selection(D33=D_FACTOR2_B))
    assert factor2_d.reason == "insufficient_axis_coverage"
    assert factor2_d.axis_resultants[1] == pytest.approx(
        0.16467828332102233, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
    assert factor2_d.axis_residuals[1] == pytest.approx(
        0.22541339102299035, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
    assert factor2_d.axis_resultants[1] > CAPTURED_AXIS1_RESULTANT
    _assert_axis1_gate_failed(factor2_d)


def test_factor2_combined_with_consistent_ab_does_not_close_the_axis1_gate():
    # The R5 hypothesis was that a factor-2 label on C or D closes the triangle.
    # The frozen solver does not support that as a gate-closing fix: after the A/B
    # conflict is removed, the factor-2 arms *lower* the resultant again.
    consistent = _solve(_selection(A19=A_CAPTURED_B, B33=B_CAPTURED_B))
    both_c = _solve(_selection(A19=A_CAPTURED_B, B33=B_CAPTURED_B, C19=C_FACTOR2_B))
    both_d = _solve(_selection(A19=A_CAPTURED_B, B33=B_CAPTURED_B, D33=D_FACTOR2_B))
    assert both_c.reason == "insufficient_axis_coverage"
    assert both_d.reason == "insufficient_axis_coverage"
    assert both_c.axis_resultants[1] == pytest.approx(
        0.018322043173470668, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
    assert both_c.axis_residuals[1] == pytest.approx(
        0.2240517208226744, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
    assert both_d.axis_resultants[1] == pytest.approx(
        0.031086391079167598, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
    assert both_d.axis_residuals[1] == pytest.approx(
        0.26492565462765694, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
    assert both_c.axis_resultants[1] < consistent.axis_resultants[1]
    assert both_d.axis_resultants[1] < consistent.axis_resultants[1]
    _assert_axis1_gate_failed(both_c)
    _assert_axis1_gate_failed(both_d)


def test_cycle_closure_defect_is_reduced_but_the_gate_never_closes():
    captured_closure, captured_defect = _cycle_closure(CAPTURED_SELECTION)
    assert captured_closure == pytest.approx(
        4.482613592937128, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
    assert captured_defect == pytest.approx(
        0.4826135929371276, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)

    consistent_closure, consistent_defect = _cycle_closure(
        _selection(A19=A_CAPTURED_B, B33=B_CAPTURED_B))
    assert consistent_closure == pytest.approx(
        -0.6622211104892352, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
    assert consistent_defect == pytest.approx(
        0.33777888951076485, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)

    _, factor2_c_defect = _cycle_closure(_selection(C19=C_FACTOR2_B))
    _, factor2_d_defect = _cycle_closure(_selection(D33=D_FACTOR2_B))
    assert factor2_c_defect == pytest.approx(
        0.11190372861822251, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
    assert factor2_d_defect == pytest.approx(
        0.19110234676679738, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
    assert factor2_c_defect < captured_defect
    assert factor2_d_defect < captured_defect

    _, both_c_defect = _cycle_closure(
        _selection(A19=A_CAPTURED_B, B33=B_CAPTURED_B, C19=C_FACTOR2_B))
    _, both_d_defect = _cycle_closure(
        _selection(A19=A_CAPTURED_B, B33=B_CAPTURED_B, D33=D_FACTOR2_B))
    assert both_c_defect == pytest.approx(
        0.032930974808140556, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
    assert both_d_defect == pytest.approx(
        0.046267643340435094, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
    # Closure improves monotonically, yet every solver arm above still fails the gate.
    assert both_c_defect < consistent_defect
    assert both_d_defect < consistent_defect
