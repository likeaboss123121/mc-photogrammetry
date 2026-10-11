"""DS-IMG-AXIS1-JOINT-ALT-SOLVER-TEST-R8 evaluation-only synthetic solver check.

**This fixture is synthetic and conditional.** It is not a real-image replay. It
drives the frozen public solver :func:`plane_graph.solve_component` from a reduced
active-edge subset rebuilt out of the four captured axis-1 readings

* ``A = image-0000`` plane ``p2`` (axis 1, side ``-1``),
* ``B = image-0001`` plane ``p3`` (axis 1, side ``+1``),
* ``C = image-0002`` plane ``p2`` (axis 1, side ``-1``),
* ``D = image-0002`` plane ``p3`` (axis 1, side ``+1``),

and the three captured active edges

* match 5:  A(``b``=0.25) -- B(``b``=0.125), relative quadrant 3,
* match 19: A(``b``=0.5)  -- C(``b``=0.25),  relative quadrant 1,
* match 33: B(``b``=3.0)  -- D(``b``=0.125), relative quadrant 2,

with the captured best-assignment quadrants ``[0, 3, 1]`` pinned through the
public ``quadrant_constraints`` argument. The four readings carry their **exact**
captured alternative tables (``b``, ``distance_blocks``, ``support``) and the
three edges carry the captured M5/M19/M33 binding rows (selected ``b``, selected
distance, side, sign, axis, relative quadrant).

Evidence provenance (never read by this module; the captured numbers are pinned
constants). ``.tmp-geom-phase-capture-r8-20261004/observation.json`` file sha256
``C3D2EE4AD3E803754D7297AB0C09B7994B17C24DE345FDA2810B5CB62888FF8``; its
``solver_failure_observation.solver_input_bindings`` record sha256
``25c8f1460311846cacfec929ec10f703ad8f47af9e8f41eeb5304bbe4f97c99a``, schema
``mc-photogrammetry-solver-input-bindings-v1``, ``input_match_count`` 38,
``active_edge_count`` 10, frames ``[image-0000, image-0001, image-0002]``,
assignments ``[0, 3, 1]``. The R8 artifact reports axis-1 resultant
``0.013942164981374504`` / residual ``0.24710226548952122`` with 6 face
observations. Reading ``R2`` (A) is ``image-0000`` plane ``p2``; ``R9`` (B),
``R13`` (C) and ``R14`` (D) are as above.

WHAT THIS CHECK CLAIMS. (1) *Control:* the captured selection (including the
captured A/B endpoint alias reuse) reproduces the R8 axis-1 resultant, residual,
observation count and per-frame translations exactly through the real solver.
(2) *Joint alternatives:* with the A/B labels made consistent (``A``=0.25 on both
edges, ``B``=0.125 on both edges), the three joint C/D label choices
``(0.5, 1.0)``, ``(1.0, 2.0)`` and ``(3.0, 2.0)`` each return exact solver metrics
that clear the frozen per-axis gate ``resultant >= 0.90`` and
``residual <= 0.15``.

WHAT IT DOES NOT CLAIM. R8 does not prove that these alternate combinations were
emitted as matcher pairs, or that they survived ``_apply_same_axis_filter``
admission, or that they would ever be selected by the full 38-edge assignment
search. The pre-same-axis admitted candidates that the filter dropped are not
retained in the artifact. A passing label combination proves *feasibility in the
retained label space*, not correctness: no truth, camera, scale or reference data
is available here, so a different label set is a different geometric
interpretation. The reduced subset contains only axis 1, so it cannot reach the
real run's ``plane_integer_residual`` reason; the frozen solver returns the
earlier typed subset reason ``insufficient_axis_coverage`` while its axis-1
diagnostics are still reproduced exactly and the per-axis gate is evaluated
separately. The unrecorded per-match score is set to a canonical synthetic 1.0;
this does not affect the result because the pinned quadrant constraints leave a
single feasible assignment, and score enters only the (unused) assignment cost.

Scope: this module reads no file, runs no image or reconstruction workload, reads
no truth/reference/camera/F3/direction/coordinate/bounds/mask/evaluator metadata,
and changes no production source, threshold, matcher, solver or gate. The
controls are evaluation evidence only, not production acceptance.
"""
from __future__ import annotations

import pytest

from mc_photogrammetry.reconstruction import lattice_planes, plane_graph

# --- Pinned provenance (constants only; this module performs no file I/O) ----------
R8_OBSERVATION_SHA256 = (
    "C3D2EE4AD3E803754D7297AB0C09B7994B17C24DE345FDA2810B5CB62888FF8")
R8_SOLVER_INPUT_BINDINGS_RECORD_SHA256 = (
    "25c8f1460311846cacfec929ec10f703ad8f47af9e8f41eeb5304bbe4f97c99a")

# --- Captured constants (pinned from the R8 artifact) ------------------------------

A_FRAME, A_SIDE = "image-0000", -1.0
B_FRAME, B_SIDE = "image-0001", 1.0
C_FRAME, C_SIDE = "image-0002", -1.0
D_FRAME, D_SIDE = "image-0002", 1.0

# Exact captured alternative rows R2 / R9 / R13 / R14, in the artifact's order:
# (blocks_per_period, distance_blocks, support).
A_ALTERNATIVES = (
    (1.0, 3.0746233309304403, 1.0),
    (0.5, 1.5373116654652201, 0.8112677931785583),
    (0.25, 0.7686558327326101, 0.6519330739974976),
)
B_ALTERNATIVES = (
    (1.0, 1.5221491724152183, 1.0),
    (0.5, 0.7610745862076091, 0.9160945415496826),
    (0.25, 0.3805372931038046, 0.7769672274589539),
    (2.0, 3.0442983448304366, 0.5287622213363647),
    (3.0, 4.5664475172456545, 0.5176740884780884),
    (0.125, 0.1902686465519023, 0.5137729644775391),
)
C_ALTERNATIVES = (
    (1.0, 1.4828394572756216, 1.0),
    (0.5, 0.7414197286378108, 0.9366976022720337),
    (0.25, 0.3707098643189054, 0.7766876220703125),
    (2.0, 2.965678914551243, 0.5781933665275574),
    (3.0, 4.448518371826865, 0.5616558790206909),
)
D_ALTERNATIVES = (
    (1.0, 2.3320899693626376, 1.0),
    (0.5, 1.1660449846813188, 0.8231529593467712),
    (2.0, 4.664179938725275, 0.7062641382217407),
    (0.25, 0.5830224923406594, 0.6354444622993469),
    (3.0, 6.996269908087913, 0.5839419960975647),
    (4.0, 9.32835987745055, 0.5452104806900024),
    (0.125, 0.2915112461703297, 0.506087064743042),
)

# Exact M5 / M19 / M33 binding rows: (reading alt index, selected b, selected
# distance, support, side, sign, axis). The alt indices address the captured
# tables above.
M5_A = (2, 0.25, 0.7686558327326101, 0.6519330739974976, -1.0, 1, 1)
M5_B = (5, 0.125, 0.1902686465519023, 0.5137729644775391, 1.0, 1, 1)
M5_QUADRANT = 3
M19_A = (1, 0.5, 1.5373116654652201, 0.8112677931785583, -1.0, 1, 1)
M19_C = (2, 0.25, 0.3707098643189054, 0.7766876220703125, -1.0, 1, 1)
M19_QUADRANT = 1
M33_B = (4, 3.0, 4.5664475172456545, 0.5176740884780884, 1.0, 1, 1)
M33_D = (6, 0.125, 0.2915112461703297, 0.506087064743042, 1.0, 1, 1)
M33_QUADRANT = 2

# Captured selection: A and B reuse their aliases across the two edges they join.
A_EDGE_B, B_EDGE_B = 0.25, 0.125
A_ALIAS_B, B_ALIAS_B = 0.5, 3.0
C_EDGE_B, D_EDGE_B = 0.25, 0.125

PINNED_QUADRANTS = {A_FRAME: 0, B_FRAME: 3, C_FRAME: 1}
QUADRANT_CONSTRAINTS = (
    (A_FRAME, B_FRAME, 3),
    (A_FRAME, C_FRAME, 1),
    (B_FRAME, C_FRAME, 2),
)

# The frozen solver's own defaults, restated so the gate under test is explicit.
MIN_RESULTANT = 0.90
INTEGER_TOLERANCE = 0.15
AMBIGUOUS_MARGIN = 0.02

CAPTURED_AXIS1_RESULTANT = 0.013942164981374504
CAPTURED_AXIS1_RESIDUAL = 0.24710226548952122
CAPTURED_AXIS1_OBSERVATIONS = 6
CAPTURED_AXIS1_TRANSLATIONS = {
    A_FRAME: 0.0,
    B_FRAME: -2.453129010263554,
    C_FRAME: 0.32760272983272826,
}

# Consistent A/B labels with joint C/D choices, with the metrics returned by the
# real frozen solver recorded before this module was written. "Consistent A/B"
# means A keeps b=0.25 on both edges and B keeps b=0.125 on both edges.
JOINT_CD_ARMS = (
    (0.5, 1.0, 0.9891764547260209, 0.02041936055568005),
    (1.0, 2.0, 0.9571873683490001, 0.040838721111360256),
    (3.0, 2.0, 0.9746878573862846, 0.03130508626448332),
)

# The artifact reports its metrics at ``compatibility_decimals = 6`` (1e-6); the
# reduced subset drives the identical three-edge least squares, so 1e-9 absolute
# is ~1000x tighter than that documented scale while absorbing BLAS round-off.
NUMERIC_REL_TOLERANCE = 1e-9
NUMERIC_ABS_TOLERANCE = 1e-12

_RECTIFICATION = lattice_planes.PlaneRectification(
    axis_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    focal=500.0, principal_point=(120.0, 90.0), centre=(0.0, 0.0))


def _reading(frame, side, alternatives):
    ordered = tuple(
        lattice_planes.DistanceAlternative(float(b), float(d), float(s))
        for b, d, s in alternatives)
    return lattice_planes.PlaneReading(
        frame=frame, axis=1, side=float(side), alternatives=ordered,
        period_pixels=60.0, phase_pixels=(0.0, 0.0), extent=1.0, samples=512,
        polygon=None, coverage_cells=8, structural_share=0.6, confidence=0.9,
        rectification=_RECTIFICATION)


def _readings():
    """The four captured readings, ordered A, B, C, D."""
    return (
        _reading(A_FRAME, A_SIDE, A_ALTERNATIVES),
        _reading(B_FRAME, B_SIDE, B_ALTERNATIVES),
        _reading(C_FRAME, C_SIDE, C_ALTERNATIVES),
        _reading(D_FRAME, D_SIDE, D_ALTERNATIVES),
    )


def _plane_index(readings, target):
    """Resolve the ``(frame, index)`` address with the solver's own order key.

    The captured 15-reading universe indexed A at ``p2``, B at ``p3`` and C/D at
    ``p2``/``p3``; this reduced four-reading subset renumbers each frame from
    zero, exactly as the R6 replay documents.
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
        left_b=float(left_b), right_b=float(right_b),
        quadrant_of_right=quadrant, psr=5.0, peak=1.0, second_peak=0.1,
        overlap_cells=64, integer_residual=None, score=1.0,
        left_plane=left_plane, right_plane=right_plane)


def _solve(a5_b, b5_b, a19_b, c19_b, b33_b, d33_b):
    readings = _readings()
    A, B, C, D = readings
    matches = (
        _match(A_FRAME, B_FRAME, _plane_index(readings, A), _plane_index(readings, B),
               M5_QUADRANT, a5_b, b5_b),
        _match(A_FRAME, C_FRAME, _plane_index(readings, A), _plane_index(readings, C),
               M19_QUADRANT, a19_b, c19_b),
        _match(B_FRAME, D_FRAME, _plane_index(readings, B), _plane_index(readings, D),
               M33_QUADRANT, b33_b, d33_b),
    )
    return plane_graph.solve_component(
        list(readings), matches, integer_tolerance=INTEGER_TOLERANCE,
        min_resultant=MIN_RESULTANT, ambiguous_margin=AMBIGUOUS_MARGIN,
        quadrant_constraints=QUADRANT_CONSTRAINTS)


def _solve_captured():
    return _solve(A_EDGE_B, B_EDGE_B, A_ALIAS_B, C_EDGE_B, B_ALIAS_B, D_EDGE_B)


def _solve_joint_cd(c_b, d_b):
    """Consistent A/B labels plus one joint C/D choice."""
    return _solve(A_EDGE_B, B_EDGE_B, A_EDGE_B, c_b, B_EDGE_B, d_b)


def _assert_axis1_gate_passed(solution):
    assert solution.axis_resultants[1] >= MIN_RESULTANT
    assert solution.axis_residuals[1] <= INTEGER_TOLERANCE


# --- Controls ----------------------------------------------------------------------


def test_captured_abcd_rows_and_m5_m19_m33_bindings_resolve_in_the_reduced_universe():
    readings = _readings()
    A, B, C, D = readings
    assert [reading.axis for reading in readings] == [1, 1, 1, 1]
    assert [reading.side for reading in readings] == [A_SIDE, B_SIDE, C_SIDE, D_SIDE]

    # Reduced-subset renumbering: captured p2/p3 collapse to per-frame 0/0/0/1.
    assert _plane_index(readings, A) == 0
    assert _plane_index(readings, B) == 0
    assert _plane_index(readings, C) == 0
    assert _plane_index(readings, D) == 1

    # Every captured M5/M19/M33 binding row resolves to its exact alternative.
    for reading, (alt_index, b, distance, support, side, sign, axis) in (
            (A, M5_A), (B, M5_B), (A, M19_A), (C, M19_C),
            (B, M33_B), (D, M33_D)):
        alternative = reading.alternatives[alt_index]
        assert alternative.blocks_per_period == pytest.approx(
            b, rel=0.0, abs=0.0)
        assert alternative.distance_blocks == pytest.approx(
            distance, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
        assert alternative.support == pytest.approx(
            support, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
        assert reading.side == side
        assert reading.axis == axis
        assert sign == 1
        assert reading.distance_for(b) == pytest.approx(
            distance, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)

    # The joint C/D labels exercised below are retained captured alternatives.
    assert {b for b, _d, _s in C_ALTERNATIVES} >= {0.5, 1.0, 3.0}
    assert {b for b, _d, _s in D_ALTERNATIVES} >= {1.0, 2.0}


def test_control_reproduces_r8_captured_axis1_metrics():
    solution = _solve_captured()
    assert dict(solution.quadrants) == PINNED_QUADRANTS
    # Only axis 1 exists in the reduced subset, so axes 0 and 2 are absent.
    assert set(solution.axis_resultants) == {1}
    assert set(solution.axis_residuals) == {1}
    assert solution.reason in plane_graph.COMPONENT_ABSTENTION_REASONS
    # Honest limitation: the one-axis subset cannot reach the real run's
    # ``plane_integer_residual``; it stops at the earlier coverage predicate.
    assert solution.reason == "insufficient_axis_coverage"
    assert solution.axis_resultants[1] == pytest.approx(
        CAPTURED_AXIS1_RESULTANT, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
    assert solution.axis_residuals[1] == pytest.approx(
        CAPTURED_AXIS1_RESIDUAL, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
    assert solution.alternatives[0].axis_observations == (
        0, CAPTURED_AXIS1_OBSERVATIONS, 0)
    for frame, expected in CAPTURED_AXIS1_TRANSLATIONS.items():
        assert solution.translations[frame][1] == pytest.approx(
            expected, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
    # The captured alias selection fails the frozen per-axis gate.
    assert solution.axis_resultants[1] < MIN_RESULTANT
    assert solution.axis_residuals[1] > INTEGER_TOLERANCE


@pytest.mark.parametrize(
    "c_b,d_b,expected_resultant,expected_residual",
    JOINT_CD_ARMS,
    ids=["C0.5-D1.0", "C1.0-D2.0", "C3.0-D2.0"],
)
def test_joint_cd_choices_with_consistent_ab_close_the_frozen_axis1_gate(
        c_b, d_b, expected_resultant, expected_residual):
    solution = _solve_joint_cd(c_b, d_b)
    assert dict(solution.quadrants) == PINNED_QUADRANTS
    assert solution.reason == "insufficient_axis_coverage"
    assert solution.alternatives[0].axis_observations == (
        0, CAPTURED_AXIS1_OBSERVATIONS, 0)
    # Recorded expected metrics, asserted through the real frozen solver.
    assert solution.axis_resultants[1] == pytest.approx(
        expected_resultant, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
    assert solution.axis_residuals[1] == pytest.approx(
        expected_residual, rel=NUMERIC_REL_TOLERANCE, abs=NUMERIC_ABS_TOLERANCE)
    # Frozen per-axis gate, asserted independently of the recorded values.
    _assert_axis1_gate_passed(solution)
