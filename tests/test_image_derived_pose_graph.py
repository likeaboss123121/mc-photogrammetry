"""Controls for the bounded M4 image-derived relative pose graph composition.

Every test injects the accepted connection adapter's declared ``matcher`` seam,
so no SIFT runs and no pixels are read: the evidence graph is produced by the
real, accepted ``build_image_derived_connections`` over a session whose only
monkeypatched piece is ``image_only.solve_image_cameras`` (mirroring
``tests/test_image_derived_connections.py``). The composer is then exercised on
that real, digest-bound graph.

The synthetic edge data encodes the verified composition convention: a solved
edge's ``relative_rotation`` maps left camera coordinates to right camera
coordinates, its ``baseline_direction`` is the unit left-to-right vector in the
left estimate's declared world frame, and ``right_yaw_class`` is the relative
90-degree yaw gauge ``k_right - k_left`` that transports a baseline into the
component anchor's declared frame.
"""
from __future__ import annotations

import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest

from mc_photogrammetry.pipeline import image_only
from mc_photogrammetry.pipeline.contracts import (
    CameraEstimate,
    CameraIntrinsics,
    CameraPose,
    CameraSolution,
)
from mc_photogrammetry.pipeline.correspondence import FrameFeatures
from mc_photogrammetry.pipeline.image_derived_connections import (
    build_image_derived_connections,
    make_connection_budgets,
)
from mc_photogrammetry.pipeline.image_derived_pose_graph import (
    POSE_GRAPH_BUDGET_EXHAUSTED,
    POSE_GRAPH_CYCLE_DIRECTION_INCONSISTENT,
    POSE_GRAPH_CYCLE_ROTATION_INCONSISTENT,
    POSE_GRAPH_INPUT_INVALID,
    POSE_GRAPH_REFUSAL_REASONS,
    POSE_GRAPH_SOURCE_INVALID,
    POSE_GRAPH_SOURCE_REFUSED,
    GaugeCameraPose,
    ImageDerivedPoseGraphError,
    PoseGraphBudgets,
    compose_image_derived_pose_graph,
    make_pose_graph_budgets,
)
from mc_photogrammetry.pipeline.image_derived_session import build_image_derived_session
from mc_photogrammetry.pipeline.joint_matching import match_feature_frames
from mc_photogrammetry.reconstruction.camera import (
    CameraModel,
    intrinsics_for,
    project_points,
)

_SQRT3_OVER_2 = math.sqrt(3.0) / 2.0


def test_partial_disconnected_views_keep_separate_component_gauges(monkeypatch):
    from test_image_derived_session import _partial_session
    session, _ = _partial_session(monkeypatch)
    graph = build_image_derived_connections(
        session, budgets=make_connection_budgets(max_pairs=4, max_total_support=128),
        matcher=lambda left_id, left_image, right_id, right_image, **kwargs: _refused_result(left_id, right_id))
    assert graph.components == (("image-0000",), ("image-0002",))
    poses = compose_image_derived_pose_graph(graph, budgets=make_pose_graph_budgets(
        max_edges=4, max_cycle_checks=8,
        rotation_tolerance_degrees=1.0, direction_tolerance=1e-9))
    assert poses.components == graph.components and poses.frame_ids == graph.frame_ids
    assert len({pose.component_id for pose in poses.poses}) == 2
    assert poses.metric_scale_known is False


# --- Session, graph and edge fixtures -----------------------------------------

def _canonical_unresolved(frame_ids):
    notes: list[str] = []
    for frame_id in frame_ids:
        notes.append(f"{frame_id}: position (vanishing points fix no translation)")
        notes.append(f"{frame_id}: scale (needs the lattice period)")
        notes.append(f"{frame_id}: yaw quadrant (recovered modulo 90 degrees)")
    return tuple(notes)


def _fake_estimate(yaw=14.0):
    return CameraEstimate(
        intrinsics=CameraIntrinsics((80, 60), (100.0, 100.0), (40.0, 30.0)),
        pose=CameraPose(position=None, yaw_degrees=yaw, pitch_degrees=3.0, roll_degrees=0.0),
        residual_pixels=0.4,
        confidence=0.0,
    )


def _gray(seed=0, shape=(60, 80)):
    return np.random.default_rng(seed).integers(0, 256, size=shape, dtype=np.uint8)


def _session(monkeypatch, frame_ids):
    estimates = {frame_id: _fake_estimate() for frame_id in frame_ids}

    def fake_solve(frames):
        return CameraSolution(
            cameras={frame_id: estimates[frame_id] for frame_id in frames},
            unresolved=_canonical_unresolved(list(frames)),
            scale_known=False,
        )

    monkeypatch.setattr(image_only, "solve_image_cameras", fake_solve)
    return build_image_derived_session(
        {frame_id: _gray(seed=index) for index, frame_id in enumerate(frame_ids)})


def _refused_result(left_id, right_id, reason="candidate_graph_empty", support=0):
    return SimpleNamespace(
        left_id=left_id, right_id=right_id, status="refused", reason=reason, solved=False,
        pose=None, right_yaw_class=None, matches=np.empty((0, 4), dtype=np.float64),
        support=support, support_floor=8, raw_keypoints=(support, support),
        candidate_edges=support, mutual_survivors=support,
        median_selected_residual_degrees=None, parallax_degrees=None,
        cheirality_fraction=None, singular_ratio=None, ambiguity_margin=None,
        hypotheses=(),
    )


def _solved_result(left_id, right_id, baseline, rotation, *, yaw_class=0, support=6):
    return SimpleNamespace(
        left_id=left_id, right_id=right_id, status="solved", reason="", solved=True,
        pose=SimpleNamespace(
            baseline_direction=tuple(float(value) for value in baseline),
            relative_rotation=np.asarray(rotation, dtype=np.float64),
            solved=True),
        right_yaw_class=yaw_class,
        matches=np.arange(4 * support, dtype=np.float64).reshape(support, 4),
        support=support, support_floor=8, raw_keypoints=(support, support),
        candidate_edges=support * 2, mutual_survivors=support,
        median_selected_residual_degrees=0.4, parallax_degrees=5.0,
        cheirality_fraction=1.0, singular_ratio=0.01, ambiguity_margin=3, hypotheses=(),
    )


def _ry(degrees):
    angle = math.radians(degrees)
    return np.asarray(
        ((math.cos(angle), 0.0, math.sin(angle)), (0.0, 1.0, 0.0),
         (-math.sin(angle), 0.0, math.cos(angle))),
        dtype=np.float64,
    )


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / float(np.linalg.norm(vector))


def _graph(monkeypatch, frame_ids, table):
    """Build a real connection graph; ``table`` maps a pair to a solved spec."""
    session = _session(monkeypatch, frame_ids)

    def matcher(left_id, left_image, right_id, right_image, *, left_camera, right_camera):
        spec = table.get((left_id, right_id))
        if spec is None:
            return _refused_result(left_id, right_id)
        baseline, rotation, yaw_class = spec
        return _solved_result(left_id, right_id, baseline, rotation, yaw_class=yaw_class)

    return build_image_derived_connections(
        session,
        budgets=make_connection_budgets(max_pairs=64, max_total_support=10 ** 9),
        matcher=matcher,
    )


def _budgets(max_edges=64, max_cycle_checks=64, rotation=1.0, direction=1e-9):
    return make_pose_graph_budgets(
        max_edges=max_edges,
        max_cycle_checks=max_cycle_checks,
        rotation_tolerance_degrees=rotation,
        direction_tolerance=direction,
    )


def _pose(result, frame_id):
    return next(pose for pose in result.poses if pose.frame_id == frame_id)


def _keys(value):
    if isinstance(value, dict):
        collected = set(value)
        for item in value.values():
            collected |= _keys(item)
        return collected
    if isinstance(value, (list, tuple)):
        collected = set()
        for item in value:
            collected |= _keys(item)
        return collected
    return set()


# --- Chain composition --------------------------------------------------------

def test_chain_composition_propagates_rotation_and_unit_position(monkeypatch):
    graph = _graph(monkeypatch, ("a", "b", "c"), {
        ("a", "b"): ((1.0, 0.0, 0.0), _ry(10.0), 0),
        ("b", "c"): ((0.0, 1.0, 0.0), _ry(20.0), 0),
    })

    result = compose_image_derived_pose_graph(graph, budgets=_budgets())

    assert result.status == "ok"
    assert result.components == (("a", "b", "c"),)
    assert result.metric_scale_known is False
    anchor = _pose(result, "a")
    assert anchor.rotation == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert anchor.position == (0.0, 0.0, 0.0)
    assert _pose(result, "b").position == pytest.approx((1.0, 0.0, 0.0))
    assert _pose(result, "b").rotation == pytest.approx(_ry(10.0))
    assert _pose(result, "c").position == pytest.approx((1.0, 1.0, 0.0))
    assert _pose(result, "c").rotation == pytest.approx(_ry(30.0))
    assert all(pose.graph_sha256 == graph.graph_sha256 for pose in result.poses)
    # Unit-edge gauge: every tree step has unit length.
    step = np.asarray(_pose(result, "c").position) - np.asarray(_pose(result, "b").position)
    assert float(np.linalg.norm(step)) == pytest.approx(1.0)


def test_child_gauge_transports_the_baseline_into_the_anchor_frame(monkeypatch):
    # Frame b's declared yaw gauge is one quadrant away from the anchor, so the
    # (b, c) baseline must be rotated by Ry90(-1) before it is added: a
    # composition that used the raw world vector would place c at (2, 0, 0).
    graph = _graph(monkeypatch, ("a", "b", "c"), {
        ("a", "b"): ((1.0, 0.0, 0.0), _ry(10.0), 1),
        ("b", "c"): ((1.0, 0.0, 0.0), _ry(20.0), 1),
    })

    result = compose_image_derived_pose_graph(graph, budgets=_budgets())

    assert result.status == "ok"
    assert _pose(result, "b").position == pytest.approx((1.0, 0.0, 0.0))
    assert _pose(result, "c").position == pytest.approx((1.0, 0.0, 1.0))
    assert _pose(result, "c").rotation == pytest.approx(_ry(30.0))


# --- Components and isolated frames ------------------------------------------

def test_disconnected_components_anchor_independently(monkeypatch):
    graph = _graph(monkeypatch, ("a", "b", "c", "d"), {
        ("a", "b"): ((1.0, 0.0, 0.0), _ry(15.0), 0),
        ("c", "d"): ((0.0, 0.0, 1.0), _ry(25.0), 0),
    })

    result = compose_image_derived_pose_graph(graph, budgets=_budgets())

    assert result.status == "ok"
    assert result.components == (("a", "b"), ("c", "d"))
    assert _pose(result, "a").component_id == 0
    assert _pose(result, "b").component_id == 0
    assert _pose(result, "c").component_id == 1
    assert _pose(result, "d").component_id == 1
    assert _pose(result, "c").position == (0.0, 0.0, 0.0)
    assert _pose(result, "c").rotation == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert _pose(result, "d").position == pytest.approx((0.0, 0.0, 1.0))
    assert _pose(result, "b").anchor_frame_id == "a"
    assert _pose(result, "d").anchor_frame_id == "c"


def test_isolated_frames_emit_identity_gauge_poses(monkeypatch):
    graph = _graph(monkeypatch, ("a", "b", "c"), {})

    result = compose_image_derived_pose_graph(graph, budgets=_budgets())
    dense = compose_image_derived_pose_graph(graph, budgets=_budgets(max_cycle_checks=0))

    assert result.status == "ok"
    assert result.components == (("a",), ("b",), ("c",))
    assert [pose.component_id for pose in result.poses] == [0, 1, 2]
    for pose in result.poses:
        assert pose.anchor_frame_id == pose.frame_id
        assert pose.rotation == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        assert pose.position == (0.0, 0.0, 0.0)
    # No solved edges means no cycles, so a zero cycle budget still admits it.
    assert dense.status == "ok"


# --- Cycle consistency --------------------------------------------------------

def _triangle(monkeypatch, *, ab, bc, ac, rotation_ab, rotation_bc, rotation_ac):
    return _graph(monkeypatch, ("a", "b", "c"), {
        ("a", "b"): (ab, rotation_ab, 0),
        ("a", "c"): (ac, rotation_ac, 0),
        ("b", "c"): (bc, rotation_bc, 0),
    })


def test_consistent_cycle_composes_without_refusal(monkeypatch):
    # Unit directions that close with equal positive lengths and rotations that
    # multiply around the loop to the identity.
    graph = _triangle(
        monkeypatch,
        ab=(1.0, 0.0, 0.0), bc=(-0.5, _SQRT3_OVER_2, 0.0),
        ac=(0.5, _SQRT3_OVER_2, 0.0),
        rotation_ab=_ry(10.0), rotation_bc=_ry(20.0), rotation_ac=_ry(30.0),
    )

    result = compose_image_derived_pose_graph(graph, budgets=_budgets())

    assert result.status == "ok"
    assert result.components == (("a", "b", "c"),)
    assert len(result.poses) == 3


def test_a_longer_consistent_cycle_uses_the_four_direction_face(monkeypatch):
    # A four-frame loop: the origin lies in the convex hull of four signed unit
    # directions, so the size-four Caratheodory face is required (one rotation
    # check plus C(4,2)+C(4,3)+C(4,4) = 11 subset checks).
    graph = _graph(monkeypatch, ("a", "b", "c", "d"), {
        ("a", "b"): ((1.0, 0.0, 0.0), _ry(10.0), 0),
        ("b", "c"): ((0.0, 1.0, 0.0), _ry(20.0), 0),
        ("c", "d"): ((-1.0, 0.0, 0.0), _ry(30.0), 0),
        ("a", "d"): ((0.0, 1.0, 0.0), _ry(60.0), 0),
    })

    admitted = compose_image_derived_pose_graph(graph, budgets=_budgets(max_cycle_checks=12))
    refused = compose_image_derived_pose_graph(graph, budgets=_budgets(max_cycle_checks=11))

    assert admitted.status == "ok", admitted.detail
    assert admitted.components == (("a", "b", "c", "d"),)
    assert refused.status == "refused"
    assert refused.reason == POSE_GRAPH_BUDGET_EXHAUSTED


def test_rotation_inconsistent_cycle_is_refused_atomically(monkeypatch):
    graph = _triangle(
        monkeypatch,
        ab=(1.0, 0.0, 0.0), bc=(-0.5, _SQRT3_OVER_2, 0.0),
        ac=(0.5, _SQRT3_OVER_2, 0.0),
        rotation_ab=_ry(10.0), rotation_bc=_ry(20.0), rotation_ac=_ry(50.0),
    )

    result = compose_image_derived_pose_graph(graph, budgets=_budgets())

    assert result.status == "refused"
    assert result.reason == POSE_GRAPH_CYCLE_ROTATION_INCONSISTENT
    assert result.reason in POSE_GRAPH_REFUSAL_REASONS
    assert result.poses == () and result.components == ()
    assert "rotation" in result.detail


def test_direction_inconsistent_cycle_is_refused_atomically(monkeypatch):
    # Rotations close, but all three signed unit directions lie in one quadrant,
    # so no positive edge lengths can close the loop.
    graph = _triangle(
        monkeypatch,
        ab=(1.0, 0.0, 0.0), bc=(0.0, 1.0, 0.0),
        ac=tuple(-_unit((0.7, 0.7, 0.0))),
        rotation_ab=_ry(10.0), rotation_bc=_ry(20.0), rotation_ac=_ry(30.0),
    )

    result = compose_image_derived_pose_graph(graph, budgets=_budgets())

    assert result.status == "refused"
    assert result.reason == POSE_GRAPH_CYCLE_DIRECTION_INCONSISTENT
    assert result.poses == () and result.components == ()
    assert "direction" in result.detail


def test_a_tight_direction_tolerance_but_consistent_cycle_is_admitted(monkeypatch):
    graph = _triangle(
        monkeypatch,
        ab=(1.0, 0.0, 0.0), bc=(-0.5, _SQRT3_OVER_2, 0.0),
        ac=(0.5, _SQRT3_OVER_2, 0.0),
        rotation_ab=_ry(10.0), rotation_bc=_ry(20.0), rotation_ac=_ry(30.0),
    )

    result = compose_image_derived_pose_graph(
        graph, budgets=_budgets(rotation=1e-9, direction=1e-12))

    assert result.status == "ok"


# --- Budgets ------------------------------------------------------------------

def test_edge_budget_refuses_before_any_composition(monkeypatch):
    graph = _graph(monkeypatch, ("a", "b", "c"), {
        ("a", "b"): ((1.0, 0.0, 0.0), _ry(10.0), 0),
        ("b", "c"): ((0.0, 1.0, 0.0), _ry(20.0), 0),
    })

    result = compose_image_derived_pose_graph(graph, budgets=_budgets(max_edges=1))

    assert result.status == "refused"
    assert result.reason == POSE_GRAPH_BUDGET_EXHAUSTED
    assert "max_edges" in result.detail
    assert result.poses == () and result.components == ()


def test_cycle_budget_refuses_atomically(monkeypatch):
    graph = _triangle(
        monkeypatch,
        ab=(1.0, 0.0, 0.0), bc=(-0.5, _SQRT3_OVER_2, 0.0),
        ac=(0.5, _SQRT3_OVER_2, 0.0),
        rotation_ab=_ry(10.0), rotation_bc=_ry(20.0), rotation_ac=_ry(30.0),
    )
    # One rotation closure plus four Caratheodory subsets need five checks.
    admitted = compose_image_derived_pose_graph(graph, budgets=_budgets(max_cycle_checks=5))
    refused = compose_image_derived_pose_graph(graph, budgets=_budgets(max_cycle_checks=4))

    assert admitted.status == "ok"
    assert refused.status == "refused"
    assert refused.reason == POSE_GRAPH_BUDGET_EXHAUSTED
    assert "max_cycle_checks" in refused.detail
    assert refused.poses == () and refused.components == ()


def test_cycle_budget_zero_still_admits_a_chain(monkeypatch):
    graph = _graph(monkeypatch, ("a", "b", "c"), {
        ("a", "b"): ((1.0, 0.0, 0.0), _ry(10.0), 0),
        ("b", "c"): ((0.0, 1.0, 0.0), _ry(20.0), 0),
    })

    result = compose_image_derived_pose_graph(graph, budgets=_budgets(max_cycle_checks=0))

    assert result.status == "ok"
    assert result.components == (("a", "b", "c"),)


def test_cycle_budget_zero_refuses_a_cycle_before_any_tree_work(monkeypatch):
    graph = _triangle(
        monkeypatch,
        ab=(1.0, 0.0, 0.0), bc=(-0.5, _SQRT3_OVER_2, 0.0),
        ac=(0.5, _SQRT3_OVER_2, 0.0),
        rotation_ab=_ry(10.0), rotation_bc=_ry(20.0), rotation_ac=_ry(30.0),
    )

    result = compose_image_derived_pose_graph(graph, budgets=_budgets(max_cycle_checks=0))

    assert result.status == "refused"
    assert result.reason == POSE_GRAPH_BUDGET_EXHAUSTED
    assert "fundamental cycle count" in result.detail
    assert result.poses == () and result.components == ()


def test_budgets_are_explicit_and_self_validating():
    with pytest.raises(ValueError):
        PoseGraphBudgets(-1, 0, 1.0, 1e-9)
    with pytest.raises(ValueError):
        PoseGraphBudgets(1, -1, 1.0, 1e-9)
    with pytest.raises(ValueError):
        PoseGraphBudgets(1, 1, -0.1, 1e-9)
    with pytest.raises(ValueError):
        PoseGraphBudgets(1, 1, 181.0, 1e-9)
    with pytest.raises(ValueError):
        PoseGraphBudgets(1, 1, 1.0, 1.5)
    with pytest.raises(ValueError):
        PoseGraphBudgets(1, 1, 1.0, 1e-9, budget_sha256="0" * 64)
    budgets = make_pose_graph_budgets(
        max_edges=3, max_cycle_checks=5, rotation_tolerance_degrees=1.0,
        direction_tolerance=1e-9)
    assert budgets.max_edges == 3 and budgets.max_cycle_checks == 5
    assert budgets.budget_sha256 is not None and len(budgets.budget_sha256) == 64
    assert budgets.to_payload()["budget_sha256"] == budgets.budget_sha256


# --- Source-binding and input refusals ---------------------------------------

def test_malformed_graph_is_a_typed_source_refusal(monkeypatch):
    graph = _graph(monkeypatch, ("a", "b"), {
        ("a", "b"): ((1.0, 0.0, 0.0), _ry(10.0), 0),
    })
    object.__setattr__(graph, "graph_sha256", "0" * 64)

    result = compose_image_derived_pose_graph(graph, budgets=_budgets(max_edges=0))

    assert result.status == "refused"
    assert result.reason == POSE_GRAPH_SOURCE_INVALID
    assert result.reason in POSE_GRAPH_REFUSAL_REASONS
    assert result.poses == () and result.components == ()


def test_tampered_evidence_record_is_a_typed_source_refusal(monkeypatch):
    graph = _graph(monkeypatch, ("a", "b"), {
        ("a", "b"): ((1.0, 0.0, 0.0), _ry(10.0), 0),
    })
    object.__setattr__(graph.evidence[0], "support", 999999)

    result = compose_image_derived_pose_graph(graph, budgets=_budgets())

    assert result.status == "refused"
    assert result.reason == POSE_GRAPH_SOURCE_INVALID


def test_a_non_so3_solved_rotation_is_refused_at_zero_budgets(monkeypatch):
    # ``diag(2, 1, 1)`` is finite and 3x3, so the accepted evidence constructor
    # and the dataclasses.replace replay both admit it; the composer's own SO(3)
    # re-validation must refuse it typed. The refusal is a source-data
    # condition, so it fires before the solved-edge budget: even with every
    # budget at zero the result is POSE_GRAPH_SOURCE_INVALID with no poses and
    # no components, never POSE_GRAPH_BUDGET_EXHAUSTED and never a raw
    # ValueError from pose construction.
    graph = _graph(monkeypatch, ("a", "b"), {
        ("a", "b"): ((1.0, 0.0, 0.0), np.diag([2.0, 1.0, 1.0]), 0),
    })
    assert graph.status == "ok"
    assert graph.evidence[0].status == "solved"

    result = compose_image_derived_pose_graph(
        graph, budgets=_budgets(
            max_edges=0, max_cycle_checks=0, rotation=0.0, direction=0.0))

    assert result.status == "refused"
    assert result.reason == POSE_GRAPH_SOURCE_INVALID
    assert result.reason in POSE_GRAPH_REFUSAL_REASONS
    assert result.poses == () and result.components == ()
    assert "relative_rotation" in result.detail
    assert "orthonormal" in result.detail


def test_a_reflection_solved_rotation_is_refused_by_the_determinant_check(monkeypatch):
    # ``diag(1, 1, -1)`` is orthonormal but has determinant -1, so it passes the
    # orthonormality test and must be caught by the explicit determinant +1
    # tolerance instead.
    graph = _graph(monkeypatch, ("a", "b"), {
        ("a", "b"): ((1.0, 0.0, 0.0), np.diag([1.0, 1.0, -1.0]), 0),
    })
    assert graph.status == "ok"

    result = compose_image_derived_pose_graph(graph, budgets=_budgets())

    assert result.status == "refused"
    assert result.reason == POSE_GRAPH_SOURCE_INVALID
    assert result.poses == () and result.components == ()
    assert "determinant +1" in result.detail


def test_the_solved_rotation_so3_tolerance_is_explicit(monkeypatch):
    # The source check is a tolerance check, not an exact-equality check: a
    # 1e-9 orthonormality defect is inside the explicit 1e-6 bound and composes,
    # while a 1e-3 defect is outside it and refuses typed.
    def perturbed(delta):
        matrix = _ry(10.0)
        matrix[0, 0] += delta
        return matrix

    admitted = compose_image_derived_pose_graph(
        _graph(monkeypatch, ("a", "b"), {
            ("a", "b"): ((1.0, 0.0, 0.0), perturbed(1e-9), 0),
        }),
        budgets=_budgets())
    refused = compose_image_derived_pose_graph(
        _graph(monkeypatch, ("a", "b"), {
            ("a", "b"): ((1.0, 0.0, 0.0), perturbed(1e-3), 0),
        }),
        budgets=_budgets())

    assert admitted.status == "ok", admitted.detail
    assert refused.status == "refused"
    assert refused.reason == POSE_GRAPH_SOURCE_INVALID
    assert refused.poses == () and refused.components == ()
    assert "orthonormal" in refused.detail


def _determinant_preserving_diagonal(epsilon):
    """A finite ``3x3`` with ``det = 1`` and Gram defect ``~2*epsilon``.

    ``A = diag(1 + e, 1 / (1 + e), 1)`` has ``det(A) = 1`` exactly, so only the
    orthonormality test can ever reject it; its ``AᵀA`` entries depart from the
    identity by ``(1 + e)^2 - 1`` and ``1 - 1/(1 + e)^2``.
    """
    return np.diag([1.0 + epsilon, 1.0 / (1.0 + epsilon), 1.0])


def test_a_near_boundary_gram_defect_just_inside_the_bound_is_admitted(monkeypatch):
    # Valid near-boundary control for the explicit 1e-6 orthonormality bound: a
    # determinant-preserving Gram defect of ~8e-7 is inside the declared bound
    # (and must stay admitted now that the comparison uses rtol=0.0).
    inside = _determinant_preserving_diagonal(4e-7)
    assert abs(float(np.linalg.det(inside)) - 1.0) < 1e-12
    assert float(np.max(np.abs(inside.T @ inside - np.eye(3)))) < 1e-6

    graph = _graph(monkeypatch, ("a", "b"), {
        ("a", "b"): ((1.0, 0.0, 0.0), inside, 0),
    })
    assert graph.status == "ok"

    result = compose_image_derived_pose_graph(graph, budgets=_budgets())

    assert result.status == "ok", result.detail
    assert result.components == (("a", "b"),)
    assert _pose(result, "b").rotation == pytest.approx(inside)


def test_a_near_boundary_gram_defect_just_outside_the_bound_refuses_at_zero_budgets(
        monkeypatch):
    # Sol's finding: ``np.allclose(A.T @ A, I, atol=_ROTATION_TOLERANCE)`` left
    # NumPy's default ``rtol=1e-5`` in force, so the effective entry-wise bound
    # was ``1e-6 + 1e-5*|1| = 1.1e-5`` and a determinant-preserving Gram defect
    # just above the documented 1e-6 was silently admitted. ``A`` here has
    # ``A[0,0] = 1 + 4e-6``, ``A[1,1] = 1/(1 + 4e-6)``: ``det(A) = 1`` exactly,
    # while ``max|AᵀA - I| = (1 + 4e-6)^2 - 1 = 8.000008e-6`` is above 1e-6 but
    # below the old 1.1e-5 slack. It must be a typed source refusal with no
    # poses and no components - even at all-zero budgets, because source replay
    # precedes every budget.
    outside = _determinant_preserving_diagonal(4e-6)
    assert abs(float(np.linalg.det(outside)) - 1.0) < 1e-12
    assert float(np.max(np.abs(outside.T @ outside - np.eye(3)))) > 1e-6

    graph = _graph(monkeypatch, ("a", "b"), {
        ("a", "b"): ((1.0, 0.0, 0.0), outside, 0),
    })
    assert graph.status == "ok"
    assert graph.evidence[0].status == "solved"

    result = compose_image_derived_pose_graph(
        graph, budgets=_budgets(
            max_edges=0, max_cycle_checks=0, rotation=0.0, direction=0.0))

    assert result.status == "refused"
    assert result.reason == POSE_GRAPH_SOURCE_INVALID
    assert result.reason in POSE_GRAPH_REFUSAL_REASONS
    assert result.poses == () and result.components == ()
    assert "relative_rotation" in result.detail
    assert "orthonormal" in result.detail


def test_a_nan_solved_rotation_is_a_typed_source_refusal(monkeypatch):
    # A non-finite rotation cannot survive the accepted evidence constructor, so
    # it is injected directly; the dataclasses.replace replay must turn it into
    # the typed source refusal rather than a raw ValueError.
    graph = _graph(monkeypatch, ("a", "b"), {
        ("a", "b"): ((1.0, 0.0, 0.0), _ry(10.0), 0),
    })
    object.__setattr__(
        graph.evidence[0], "relative_rotation", ((float("nan"), 0.0, 0.0),
                                                 (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))

    result = compose_image_derived_pose_graph(graph, budgets=_budgets(max_edges=0))

    assert result.status == "refused"
    assert result.reason == POSE_GRAPH_SOURCE_INVALID
    assert result.poses == () and result.components == ()


def test_a_refused_source_graph_is_a_typed_source_refusal(monkeypatch):
    session = _session(monkeypatch, ("a", "b"))
    refused = build_image_derived_connections(
        session, budgets=make_connection_budgets(max_pairs=0, max_total_support=0),
        matcher=lambda *args, **kwargs: _refused_result("a", "b"))
    assert refused.status == "refused"
    called = []

    result = compose_image_derived_pose_graph(refused, budgets=_budgets())

    assert result.status == "refused"
    assert result.reason == POSE_GRAPH_SOURCE_REFUSED
    assert result.poses == () and result.components == ()
    assert called == []


def test_non_graph_inputs_raise_a_typed_input_error():
    with pytest.raises(ImageDerivedPoseGraphError) as info:
        compose_image_derived_pose_graph("not a graph", budgets=_budgets())
    assert info.value.reason == POSE_GRAPH_INPUT_INVALID
    with pytest.raises(ImageDerivedPoseGraphError) as info:
        compose_image_derived_pose_graph(object(), budgets=_budgets())
    assert info.value.reason == POSE_GRAPH_INPUT_INVALID


def test_a_non_budget_argument_raises_a_typed_input_error(monkeypatch):
    graph = _graph(monkeypatch, ("a", "b"), {
        ("a", "b"): ((1.0, 0.0, 0.0), _ry(10.0), 0),
    })
    with pytest.raises(ImageDerivedPoseGraphError) as info:
        compose_image_derived_pose_graph(graph, budgets="not budgets")
    assert info.value.reason == POSE_GRAPH_INPUT_INVALID


# --- Determinism, serialization and hard invariants --------------------------

def test_deterministic_digests_and_records(monkeypatch):
    table = {
        ("a", "b"): ((1.0, 0.0, 0.0), _ry(10.0), 0),
        ("b", "c"): ((0.0, 1.0, 0.0), _ry(20.0), 1),
    }
    first_graph = _graph(monkeypatch, ("a", "b", "c"), table)
    second_graph = _graph(monkeypatch, ("a", "b", "c"), table)
    assert first_graph.graph_sha256 == second_graph.graph_sha256

    first = compose_image_derived_pose_graph(first_graph, budgets=_budgets())
    again = compose_image_derived_pose_graph(first_graph, budgets=_budgets())
    second = compose_image_derived_pose_graph(second_graph, budgets=_budgets())

    assert first.pose_graph_sha256 == again.pose_graph_sha256 == second.pose_graph_sha256
    assert first.to_record() == second.to_record()
    assert first.to_record()["pose_graph_sha256"] == first.pose_graph_sha256
    assert [pose.to_payload() for pose in first.poses] == [
        pose.to_payload() for pose in second.poses]


def test_the_result_carries_no_scale_or_world_geometry(monkeypatch):
    graph = _graph(monkeypatch, ("a", "b"), {
        ("a", "b"): ((1.0, 0.0, 0.0), _ry(10.0), 0),
    })

    result = compose_image_derived_pose_graph(graph, budgets=_budgets())
    record = result.to_record()

    assert result.metric_scale_known is False
    assert record["metric_scale_known"] is False
    assert result.unresolved == graph.unresolved
    assert result.graph_sha256 == graph.graph_sha256
    assert result.session_sha256 == graph.session_sha256
    keys = _keys(record)
    forbidden = {
        "image", "pixels", "array", "pixel_data", "world", "world_position",
        "bounds", "bound", "transform", "scene", "block", "block_volume",
        "volume", "face", "face_observation", "observation", "translation",
        "translation_magnitude", "manifest", "litematic", "scale", "metric_scale",
        "export", "cli",
    }
    assert not keys & forbidden
    for pose in result.poses:
        assert all(math.isfinite(value) for value in pose.position)
        assert len(pose.position) == 3


def test_the_pose_constructor_refuses_a_non_rotation_and_a_non_pose_payload():
    with pytest.raises(ValueError):
        GaugeCameraPose(
            frame_id="a", component_id=0, anchor_frame_id="a",
            rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 2.0)),
            position=(0.0, 0.0, 0.0), graph_sha256="0" * 64)
    with pytest.raises(ValueError):
        GaugeCameraPose(
            frame_id="a", component_id=0, anchor_frame_id="a",
            rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            position=(0.0, float("nan"), 0.0), graph_sha256="0" * 64)


# --- Real-matcher binding -----------------------------------------------------

_WIDTH, _HEIGHT = 1280, 720
_GRID = tuple(
    (-9.0, float(y), float(z))
    for y in np.linspace(-6.0, 12.0, 6)
    for z in np.linspace(-9.0, 9.0, 6)
)


def _camera(position, yaw, pitch):
    return CameraModel(position, yaw, pitch, eye_height=0.0)


def _estimate(camera):
    return CameraEstimate(
        intrinsics=intrinsics_for(camera, _WIDTH, _HEIGHT),
        pose=CameraPose(
            position=None,
            yaw_degrees=float(camera.yaw_degrees % 90.0),
            pitch_degrees=float(camera.pitch_degrees),
            roll_degrees=float(camera.roll_degrees),
        ),
    )


def _features(frame_id, camera, descriptors):
    pixels = project_points(_GRID, camera, width=_WIDTH, height=_HEIGHT)
    return FrameFeatures(
        frame_id=frame_id,
        points=np.asarray([[pixel[0], pixel[1]] for pixel in pixels], dtype=np.float64),
        descriptors=descriptors, width=_WIDTH, height=_HEIGHT,
        usable_pixels=_WIDTH * _HEIGHT,
    )


def test_a_real_matcher_triangle_binds_the_yaw_gauge_transport(monkeypatch):
    """The shipped matcher's mixed per-frame yaw gauges compose exactly.

    Cameras ``a``, ``b`` and ``c`` declare yaw representatives 85, 5 and 15
    degrees (true yaws 85, 95 and 105), so frames ``b`` and ``c`` sit one
    quadrant away from the anchor. The shipped matcher must solve all three
    pairs, and the triangle must close exactly only when the recorded
    ``right_yaw_class`` gauges transport the baselines into the anchor frame.
    """
    cameras = {
        "a": _camera((-1.0, 3.0, 0.0), 85.0, 0.0),
        "b": _camera((3.0, 3.0, 1.0), 95.0, 0.0),
        "c": _camera((7.0, 3.0, -2.0), 105.0, 0.0),
    }
    estimates = {frame_id: _estimate(camera) for frame_id, camera in cameras.items()}
    descriptors = np.random.default_rng(7).normal(size=(len(_GRID), 16))
    features = {
        frame_id: _features(frame_id, camera, descriptors)
        for frame_id, camera in cameras.items()
    }

    def fake_solve(frames):
        return CameraSolution(
            cameras={frame_id: estimates[frame_id] for frame_id in frames},
            unresolved=_canonical_unresolved(list(frames)), scale_known=False)

    monkeypatch.setattr(image_only, "solve_image_cameras", fake_solve)
    session = build_image_derived_session(
        {frame_id: _gray(seed=index)
         for index, frame_id in enumerate(("a", "b", "c"))})

    def matcher(left_id, left_image, right_id, right_image, *, left_camera, right_camera):
        return match_feature_frames(
            features[left_id], features[right_id], left_camera, right_camera)

    graph = build_image_derived_connections(
        session,
        budgets=make_connection_budgets(max_pairs=16, max_total_support=10 ** 9),
        matcher=matcher,
    )
    assert graph.status == "ok"
    assert all(record.status == "solved" for record in graph.evidence)
    assert [record.right_yaw_class for record in graph.evidence] == [1, 1, 0]

    result = compose_image_derived_pose_graph(
        graph,
        budgets=make_pose_graph_budgets(
            max_edges=16, max_cycle_checks=32,
            rotation_tolerance_degrees=1e-9, direction_tolerance=1e-12),
    )

    assert result.status == "ok", result.detail
    assert result.components == (("a", "b", "c"),)
    assert _pose(result, "a").position == (0.0, 0.0, 0.0)
    assert float(np.linalg.norm(_pose(result, "b").position)) == pytest.approx(1.0)
    assert float(np.linalg.norm(_pose(result, "c").position)) == pytest.approx(1.0)
