"""Controls for the bounded M4 connected-camera evidence adapter.

The real matcher's expensive half is SIFT; every test here therefore injects the
adapter's declared ``matcher`` seam. Two kinds of injected matcher are used:

* synthetic stubs that return :class:`types.SimpleNamespace` records shaped
  exactly like :class:`~mc_photogrammetry.pipeline.joint_matching.JointPairResult`
  (to drive budgets, ordering and digest controls cheaply); and
* the public :func:`~mc_photogrammetry.pipeline.joint_matching.match_feature_frames`
  seam on a synthetic projected scene (to prove the adapter binds to the real
  ``JointPairResult`` contract, including its unit baseline direction and typed
  refusals, with no SIFT and no pixels read).

Sessions are produced by the accepted ``build_image_derived_session`` with only
``image_only.solve_image_cameras`` monkeypatched, mirroring
``tests/test_image_derived_session.py``.
"""
from __future__ import annotations

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
    CONNECTIONS_BUDGET_EXHAUSTED,
    CONNECTIONS_INPUT_INVALID,
    CONNECTIONS_INVALID_BASELINE_DIRECTION,
    CONNECTIONS_MATCHER_MISMATCH,
    CONNECTIONS_NO_PIXELS,
    CONNECTIONS_PAIR_REFUSAL_REASONS,
    CONNECTIONS_REFUSAL_REASONS,
    CONNECTIONS_SESSION_MISMATCH,
    ConnectionBudgets,
    ImageDerivedConnectionError,
    build_image_derived_connections,
    make_connection_budgets,
)
from mc_photogrammetry.pipeline.image_derived_session import (
    build_image_derived_session,
    image_derived_session_from_record,
    image_derived_session_to_record,
    image_sha256,
)
from mc_photogrammetry.pipeline.joint_matching import (
    JOINT_REFUSAL_REASONS,
    match_feature_frames,
)
from mc_photogrammetry.reconstruction.camera import (
    CameraModel,
    intrinsics_for,
    project_points,
)

WIDTH, HEIGHT = 1280, 720
LEFT_POSITION = (0.0, 3.0, 0.0)
RIGHT_POSITION = (2.5, 3.0, 1.0)
LEFT_YAW, LEFT_PITCH = 85.0, 3.0
RIGHT_YAW, RIGHT_PITCH = 95.0, -2.0
RIGHT_YAW_CLASS = 1

_POINTS = (
    (-9.0, 3.0, 0.5), (-8.0, 2.0, 1.5), (-10.0, 4.0, -0.5), (-9.5, 1.5, 2.0),
    (-8.5, 5.0, 0.0), (-11.0, 3.5, 1.0), (-7.5, 2.5, -1.0), (-10.5, 2.0, 2.5),
    (-9.0, 5.5, -1.5), (-12.0, 4.5, 0.5), (-8.0, 3.8, 2.8), (-10.0, 1.0, -0.8),
    (-13.0, 3.0, 1.5), (-7.0, 3.2, 0.8), (-11.5, 2.8, -1.8), (-9.8, 4.8, 2.2),
    (-8.8, 1.2, 0.2), (-12.5, 2.2, 2.0), (-10.8, 5.2, -1.2), (-7.8, 4.2, -1.8),
)


# --- Session and matcher fixtures --------------------------------------------

def _canonical_unresolved(frame_ids):
    notes: list[str] = []
    for frame_id in frame_ids:
        notes.append(f"{frame_id}: position (vanishing points fix no translation)")
        notes.append(f"{frame_id}: scale (needs the lattice period)")
        notes.append(f"{frame_id}: yaw quadrant (recovered modulo 90 degrees)")
    return tuple(notes)


def _fake_estimate(image_size=(80, 60), yaw=14.0):
    return CameraEstimate(
        intrinsics=CameraIntrinsics(
            tuple(image_size),
            (100.0, 100.0),
            (image_size[0] / 2.0, image_size[1] / 2.0),
        ),
        pose=CameraPose(position=None, yaw_degrees=yaw, pitch_degrees=3.0, roll_degrees=0.0),
        residual_pixels=0.4,
        confidence=0.0,
    )


def _gray(seed=0, shape=(60, 80)):
    return np.random.default_rng(seed).integers(0, 256, size=shape, dtype=np.uint8)


def _rgb(seed=0, shape=(60, 80)):
    return np.random.default_rng(seed).integers(0, 256, size=(*shape, 3), dtype=np.uint8)


def _build_session(monkeypatch, frame_ids, estimates, *, images=None):
    def fake_solve(frames):
        return CameraSolution(
            cameras={frame_id: estimates[frame_id] for frame_id in frames},
            unresolved=_canonical_unresolved(list(frames)),
            scale_known=False,
        )

    monkeypatch.setattr(image_only, "solve_image_cameras", fake_solve)
    if images is None:
        images = {frame_id: _gray(seed=index)
                  for index, frame_id in enumerate(frame_ids)}
    return build_image_derived_session({frame_id: images[frame_id] for frame_id in frame_ids})


def _simple_session(monkeypatch, frame_ids):
    estimates = {frame_id: _fake_estimate() for frame_id in frame_ids}
    return _build_session(monkeypatch, frame_ids, estimates)


def test_partial_connections_use_only_admitted_pair_and_replay_excluded_pixels(monkeypatch):
    from test_image_derived_session import _partial_session
    session, _ = _partial_session(monkeypatch)
    calls = []
    def matcher(left_id, left_image, right_id, right_image, **kwargs):
        calls.append((left_id, right_id))
        return SimpleNamespace(left_id=left_id, right_id=right_id, status="refused", reason="candidate_graph_empty", solved=False,
                               pose=None, right_yaw_class=None, matches=np.empty((0, 4)), support=0, support_floor=8,
                               raw_keypoints=(0, 0), candidate_edges=0, mutual_survivors=0,
                               median_selected_residual_degrees=None, parallax_degrees=None,
                               cheirality_fraction=None, singular_ratio=None, ambiguity_margin=None, hypotheses=())
    graph = build_image_derived_connections(session, budgets=_budgets(), matcher=matcher)
    assert calls == [("image-0000", "image-0002")] and graph.frame_ids == ("image-0000", "image-0002")
    excluded = session.input_frames[1]
    object.__setattr__(excluded, "image", np.frombuffer(np.ones(excluded.image_shape, dtype=np.uint8).tobytes(), dtype=np.uint8).reshape(excluded.image_shape))
    calls.clear()
    refused = build_image_derived_connections(session, budgets=_budgets(), matcher=matcher)
    assert refused.status == "refused" and calls == []


def _budgets(max_pairs=64, max_total_support=10 ** 9):
    return make_connection_budgets(max_pairs=max_pairs, max_total_support=max_total_support)


def _refused_result(left_id, right_id, *, reason="candidate_graph_empty", support=0,
                    matches=None):
    if matches is None:
        matches = np.empty((0, 4), dtype=np.float64)
    return SimpleNamespace(
        left_id=left_id, right_id=right_id, status="refused", reason=reason, solved=False,
        pose=None, right_yaw_class=None, matches=matches, support=support, support_floor=8,
        raw_keypoints=(support, support), candidate_edges=support, mutual_survivors=support,
        median_selected_residual_degrees=None, parallax_degrees=None,
        cheirality_fraction=None, singular_ratio=None, ambiguity_margin=None,
        hypotheses=(),
    )


def _solved_result(left_id, right_id, *, support=10, baseline=(1.0, 0.0, 0.0),
                   rotation=None, yaw_class=1):
    rotation = np.eye(3) if rotation is None else rotation
    matches = np.arange(4 * support, dtype=np.float64).reshape(support, 4)
    return SimpleNamespace(
        left_id=left_id, right_id=right_id, status="solved", reason="", solved=True,
        pose=SimpleNamespace(
            baseline_direction=tuple(float(value) for value in baseline),
            relative_rotation=rotation, solved=True),
        right_yaw_class=yaw_class, matches=matches, support=support, support_floor=8,
        raw_keypoints=(support, support), candidate_edges=support * 2,
        mutual_survivors=support, median_selected_residual_degrees=0.4,
        parallax_degrees=5.0, cheirality_fraction=1.0, singular_ratio=0.01,
        ambiguity_margin=3, hypotheses=(),
    )


def _hypothesis(**overrides):
    values = dict(
        yaw_class=1, support=4, spatial_bins=3,
        median_selected_residual_degrees=0.5, parallax_degrees=4.0,
        cheirality_fraction=1.0, singular_ratio=0.02,
        passed=True, refusal="",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _matcher_returning(result):
    def matcher(left_id, left_image, right_id, right_image, *, left_camera, right_camera):
        return result

    return matcher


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


# --- Synthetic projected scene (real matcher binding, no SIFT) ---------------

def _camera(position, yaw, pitch, *, intrinsics=None):
    return CameraModel(position, yaw, pitch, eye_height=0.0, intrinsics=intrinsics)


def _estimate(camera):
    return CameraEstimate(
        intrinsics=intrinsics_for(camera, WIDTH, HEIGHT),
        pose=CameraPose(
            position=None,
            yaw_degrees=float(camera.yaw_degrees % 90.0),
            pitch_degrees=float(camera.pitch_degrees),
            roll_degrees=float(camera.roll_degrees),
        ),
    )


def _features(name, pixels, descriptors):
    return FrameFeatures(
        frame_id=name, points=np.asarray(pixels, dtype=np.float64),
        descriptors=np.asarray(descriptors, dtype=np.float64),
        width=WIDTH, height=HEIGHT, usable_pixels=WIDTH * HEIGHT,
    )


def _scene():
    left_camera = _camera(LEFT_POSITION, LEFT_YAW, LEFT_PITCH)
    right_camera = _camera(RIGHT_POSITION, RIGHT_YAW, RIGHT_PITCH)
    left_projected = project_points(_POINTS, left_camera, width=WIDTH, height=HEIGHT)
    right_projected = project_points(_POINTS, right_camera, width=WIDTH, height=HEIGHT)
    left_pixels = np.asarray([[p[0], p[1]] for p in left_projected], dtype=np.float64)
    right_pixels = np.asarray([[p[0], p[1]] for p in right_projected], dtype=np.float64)
    descriptors = np.random.default_rng(3).normal(size=(len(left_pixels), 16))
    return (
        _features("a", left_pixels, descriptors),
        _features("b", right_pixels, descriptors),
        _estimate(left_camera),
        _estimate(right_camera),
    )


# --- Pair ordering, delegation and immutability ------------------------------

def test_every_unordered_pair_is_delegated_once_in_session_order(monkeypatch):
    frame_ids = ("a", "b", "c", "d")
    session = _simple_session(monkeypatch, frame_ids)
    calls: list[tuple] = []

    def matcher(left_id, left_image, right_id, right_image, *, left_camera, right_camera):
        calls.append((left_id, right_id, left_image, right_image))
        return _refused_result(left_id, right_id)

    graph = build_image_derived_connections(session, budgets=_budgets(), matcher=matcher)

    assert graph.status == "ok"
    assert graph.reason == ""
    assert graph.pair_count == 6
    assert graph.frame_ids == frame_ids
    assert [(left, right) for left, right, _l, _r in calls] == [
        ("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")]
    assert tuple((record.left_id, record.right_id) for record in graph.evidence) == tuple(
        (left, right) for left, right, _l, _r in calls)
    by_id = {frame.frame_id: frame for frame in session.frames}
    for left_id, right_id, left_image, right_image in calls:
        assert left_image is by_id[left_id].image
        assert right_image is by_id[right_id].image
        assert left_image.flags.writeable is False
        assert right_image.flags.writeable is False
    # No pair solved, so every frame is its own component.
    assert graph.components == (("a",), ("b",), ("c",), ("d",))


def test_the_adapter_never_mutates_the_session_pixels(monkeypatch):
    frame_ids = ("a", "b")
    session = _simple_session(monkeypatch, frame_ids)
    digests_before = tuple(frame.image_sha256 for frame in session.frames)

    def matcher(left_id, left_image, right_id, right_image, *, left_camera, right_camera):
        hostile = left_image.copy()
        hostile[:] = 255
        del hostile
        return _refused_result(left_id, right_id)

    build_image_derived_connections(session, budgets=_budgets(), matcher=matcher)

    assert all(frame.image.flags.writeable is False for frame in session.frames)
    assert tuple(frame.image_sha256 for frame in session.frames) == digests_before


# --- Real matcher binding: unit vector, rotation, typed refusal --------------

def test_a_real_solved_pair_records_the_unit_baseline_verbatim(monkeypatch):
    left, right, left_estimate, right_estimate = _scene()
    session = _build_session(
        monkeypatch, ("a", "b"), {"a": left_estimate, "b": right_estimate})
    expected = match_feature_frames(left, right, left_estimate, right_estimate)

    def matcher(left_id, left_image, right_id, right_image, *, left_camera, right_camera):
        return match_feature_frames(left, right, left_camera, right_camera)

    graph = build_image_derived_connections(session, budgets=_budgets(), matcher=matcher)

    assert expected.solved is True
    record = graph.evidence[0]
    assert record.status == "solved"
    assert record.reason == ""
    assert record.right_yaw_class == RIGHT_YAW_CLASS == expected.right_yaw_class
    assert record.baseline_direction == tuple(expected.pose.baseline_direction)
    assert math.isclose(float(np.linalg.norm(record.baseline_direction)), 1.0, abs_tol=1e-12)
    assert record.relative_rotation == tuple(
        tuple(float(value) for value in row) for row in expected.pose.relative_rotation)
    assert record.support == expected.support
    assert record.match_count == int(expected.matches.shape[0])
    assert len(record.matches_sha256) == 64


def test_a_real_refused_pair_keeps_the_matchers_typed_reason(monkeypatch):
    left, right, left_estimate, right_estimate = _scene()
    session = _build_session(
        monkeypatch, ("a", "b"), {"a": left_estimate, "b": right_estimate})
    few_left = _features("a", left.points[:5], left.descriptors[:5])
    few_right = _features("b", right.points[:5], right.descriptors[:5])

    def matcher(left_id, left_image, right_id, right_image, *, left_camera, right_camera):
        return match_feature_frames(few_left, few_right, left_camera, right_camera)

    graph = build_image_derived_connections(session, budgets=_budgets(), matcher=matcher)

    record = graph.evidence[0]
    assert graph.status == "ok"
    assert record.status == "refused"
    assert record.reason in JOINT_REFUSAL_REASONS
    assert record.baseline_direction is None
    assert record.relative_rotation is None
    assert record.right_yaw_class is None


def test_typed_abstentions_are_passed_through_unchanged(monkeypatch):
    session = _simple_session(monkeypatch, ("a", "b"))

    def matcher(left_id, left_image, right_id, right_image, *, left_camera, right_camera):
        return _refused_result(left_id, right_id, reason="cheirality_failed")

    graph = build_image_derived_connections(session, budgets=_budgets(), matcher=matcher)

    record = graph.evidence[0]
    assert record.status == "refused"
    assert record.reason == "cheirality_failed"
    assert record.reason in CONNECTIONS_PAIR_REFUSAL_REASONS
    assert graph.status == "ok"


# --- Digest determinism -------------------------------------------------------

def test_evidence_and_graph_digests_are_deterministic(monkeypatch):
    frame_ids = ("a", "b", "c")
    images = {frame_id: _gray(seed=index)
              for index, frame_id in enumerate(frame_ids)}
    estimates = {frame_id: _fake_estimate() for frame_id in frame_ids}

    first_session = _build_session(monkeypatch, frame_ids, estimates, images=images)
    second_session = _build_session(monkeypatch, frame_ids, estimates, images=images)
    assert first_session.session_sha256 == second_session.session_sha256

    def matcher(left_id, left_image, right_id, right_image, *, left_camera, right_camera):
        if (left_id, right_id) == ("a", "b"):
            return _solved_result(left_id, right_id, support=4)
        return _refused_result(left_id, right_id, reason="low_parallax")

    first = build_image_derived_connections(first_session, budgets=_budgets(), matcher=matcher)
    again = build_image_derived_connections(first_session, budgets=_budgets(), matcher=matcher)
    second = build_image_derived_connections(second_session, budgets=_budgets(), matcher=matcher)

    assert first.graph_sha256 == again.graph_sha256 == second.graph_sha256
    assert [record.evidence_sha256 for record in first.evidence] == [
        record.evidence_sha256 for record in second.evidence]
    assert all(record.session_sha256 == first_session.session_sha256
               for record in first.evidence)
    assert first.to_record()["graph_sha256"] == first.graph_sha256


# --- Budgets: atomic, before any partial result ------------------------------

def test_pair_budget_refuses_before_any_matcher_call(monkeypatch):
    session = _simple_session(monkeypatch, ("a", "b", "c"))
    calls: list[tuple] = []

    def matcher(*args, **kwargs):
        calls.append(args)
        raise AssertionError("the matcher must not run when the pair budget refuses")

    graph = build_image_derived_connections(
        session, budgets=_budgets(max_pairs=2), matcher=matcher)

    assert graph.status == "refused"
    assert graph.reason == CONNECTIONS_BUDGET_EXHAUSTED
    assert graph.evidence == ()
    assert graph.components == ()
    assert graph.pair_count == 3
    assert graph.graph_sha256 is not None
    assert calls == []


def test_total_budget_discards_partial_evidence(monkeypatch):
    session = _simple_session(monkeypatch, ("a", "b", "c"))
    support = {("a", "b"): 2, ("a", "c"): 10}
    calls: list[tuple] = []

    def matcher(left_id, left_image, right_id, right_image, *, left_camera, right_camera):
        calls.append((left_id, right_id))
        return _solved_result(left_id, right_id, support=support[(left_id, right_id)])

    graph = build_image_derived_connections(
        session, budgets=_budgets(max_total_support=5), matcher=matcher)

    assert graph.status == "refused"
    assert graph.reason == CONNECTIONS_BUDGET_EXHAUSTED
    assert "max_total_support" in graph.detail
    assert graph.evidence == ()
    assert graph.components == ()
    # The first pair fit and the second crossed the budget; no prefix survives.
    assert calls == [("a", "b"), ("a", "c")]


def test_budgets_are_explicit_and_self_validating():
    with pytest.raises(ValueError):
        ConnectionBudgets(max_pairs=-1, max_total_support=0)
    with pytest.raises(ValueError):
        ConnectionBudgets(max_pairs=1, max_total_support=-1)
    with pytest.raises(ValueError):
        ConnectionBudgets(max_pairs=3, max_total_support=9, budget_sha256="0" * 64)
    budgets = make_connection_budgets(max_pairs=3, max_total_support=9)
    assert budgets.max_pairs == 3 and budgets.max_total_support == 9
    assert budgets.budget_sha256 is not None and len(budgets.budget_sha256) == 64


# --- Source-binding refusals -------------------------------------------------

def test_session_rebuilt_from_a_record_without_pixels_is_refused(monkeypatch):
    session = _simple_session(monkeypatch, ("a", "b"))
    restored = image_derived_session_from_record(image_derived_session_to_record(session))
    assert restored.has_pixels is False
    calls: list[tuple] = []

    def matcher(*args, **kwargs):
        calls.append(args)
        raise AssertionError("the matcher must not run without pixels")

    graph = build_image_derived_connections(restored, budgets=_budgets(), matcher=matcher)

    assert graph.status == "refused"
    assert graph.reason == CONNECTIONS_NO_PIXELS
    assert graph.evidence == ()
    assert graph.components == ()
    assert calls == []


def test_tampered_session_digest_refuses_before_budget(monkeypatch):
    session = _simple_session(monkeypatch, ("a", "b"))
    object.__setattr__(session, "session_sha256", "0" * 64)
    calls: list[tuple] = []

    def matcher(*args, **kwargs):
        calls.append(args)
        raise AssertionError("the matcher must not run on a mismatched session")

    graph = build_image_derived_connections(
        session, budgets=_budgets(max_pairs=0, max_total_support=0), matcher=matcher)

    assert graph.status == "refused"
    assert graph.reason == CONNECTIONS_SESSION_MISMATCH
    assert graph.reason in CONNECTIONS_REFUSAL_REASONS
    assert graph.evidence == () and graph.components == ()
    assert calls == []


def test_a_swapped_numpy_owned_frame_buffer_is_refused_before_any_matcher_call(monkeypatch):
    session = _simple_session(monkeypatch, ("a", "b"))
    frame = session.frames[0]
    # Byte-identical pixels, same shape and digest, but numpy-owned and therefore
    # reopenable: this is exactly the swap the digest check alone cannot see.
    swapped = np.array(frame.image, copy=True)
    swapped.setflags(write=False)
    assert swapped.flags.writeable is False
    assert swapped.base is None
    assert image_sha256(swapped) == frame.image_sha256
    object.__setattr__(frame, "image", swapped)
    calls: list[tuple] = []

    def matcher(*args, **kwargs):
        calls.append(args)
        raise AssertionError("the matcher must not run on a reopenable buffer")

    graph = build_image_derived_connections(
        session, budgets=_budgets(max_pairs=0, max_total_support=0), matcher=matcher)

    assert graph.status == "refused"
    assert graph.reason == CONNECTIONS_SESSION_MISMATCH
    assert graph.reason in CONNECTIONS_REFUSAL_REASONS
    assert graph.evidence == () and graph.components == ()
    assert calls == []

    # Prove the swapped buffer really is reopenable and mutable: an adversarial
    # caller could have mutated the same bytes the matcher would have consumed.
    swapped.setflags(write=True)
    swapped[0, 0] = (int(swapped[0, 0]) + 1) % 256
    assert image_sha256(swapped) != frame.image_sha256


def test_a_dropped_unresolved_fact_is_a_session_mismatch(monkeypatch):
    session = _simple_session(monkeypatch, ("a", "b"))
    object.__setattr__(session, "unresolved", ())

    def matcher(*args, **kwargs):
        raise AssertionError("the matcher must not run on a mismatched session")

    graph = build_image_derived_connections(session, budgets=_budgets(), matcher=matcher)

    assert graph.status == "refused"
    assert graph.reason == CONNECTIONS_SESSION_MISMATCH


def test_an_ok_graph_preserves_the_unresolved_facts_and_scale_claim(monkeypatch):
    session = _simple_session(monkeypatch, ("a", "b"))

    def matcher(left_id, left_image, right_id, right_image, *, left_camera, right_camera):
        return _refused_result(left_id, right_id)

    graph = build_image_derived_connections(session, budgets=_budgets(), matcher=matcher)

    assert graph.status == "ok"
    assert graph.unresolved == session.unresolved
    assert graph.scale_known is False


# --- Components and unit-vector guard ----------------------------------------

def test_solved_connected_components_follow_the_solved_edges(monkeypatch):
    frame_ids = ("a", "b", "c", "d")
    session = _simple_session(monkeypatch, frame_ids)
    solved_pairs = {("a", "b"), ("b", "c")}

    def matcher(left_id, left_image, right_id, right_image, *, left_camera, right_camera):
        if (left_id, right_id) in solved_pairs:
            return _solved_result(left_id, right_id, support=3)
        return _refused_result(left_id, right_id)

    graph = build_image_derived_connections(session, budgets=_budgets(), matcher=matcher)

    assert graph.status == "ok"
    assert graph.components == (("a", "b", "c"), ("d",))
    assert {record.status for record in graph.evidence} == {"solved", "refused"}


def test_a_solved_non_unit_baseline_is_a_typed_pair_refusal(monkeypatch):
    session = _simple_session(monkeypatch, ("a", "b"))

    def matcher(left_id, left_image, right_id, right_image, *, left_camera, right_camera):
        return _solved_result(left_id, right_id, baseline=(2.0, 0.0, 0.0))

    graph = build_image_derived_connections(session, budgets=_budgets(), matcher=matcher)

    record = graph.evidence[0]
    assert record.status == "refused"
    assert record.reason == CONNECTIONS_INVALID_BASELINE_DIRECTION
    assert record.baseline_direction is None
    assert record.relative_rotation is None
    assert record.right_yaw_class is None
    assert graph.components == (("a",), ("b",))


# --- Input and matcher contract errors ---------------------------------------

def test_a_non_session_input_raises_a_typed_input_error():
    with pytest.raises(ImageDerivedConnectionError) as info:
        build_image_derived_connections("not a session", budgets=_budgets())
    assert info.value.reason == CONNECTIONS_INPUT_INVALID


def test_a_mismatched_matcher_pair_raises_a_contract_error(monkeypatch):
    session = _simple_session(monkeypatch, ("a", "b"))

    def matcher(left_id, left_image, right_id, right_image, *, left_camera, right_camera):
        return _refused_result("elsewhere", right_id)

    with pytest.raises(ImageDerivedConnectionError) as info:
        build_image_derived_connections(session, budgets=_budgets(), matcher=matcher)
    assert info.value.reason == CONNECTIONS_MATCHER_MISMATCH


def test_an_untyped_matcher_refusal_is_a_contract_error(monkeypatch):
    session = _simple_session(monkeypatch, ("a", "b"))

    def matcher(left_id, left_image, right_id, right_image, *, left_camera, right_camera):
        return _refused_result(left_id, right_id, reason="made_up_reason")

    with pytest.raises(ImageDerivedConnectionError) as info:
        build_image_derived_connections(session, budgets=_budgets(), matcher=matcher)
    assert info.value.reason == CONNECTIONS_MATCHER_MISMATCH


def test_a_malformed_matcher_hypothesis_is_a_typed_contract_error(monkeypatch):
    session = _simple_session(monkeypatch, ("a", "b"))
    malformed = (
        _hypothesis(yaw_class=7),          # outside the 0..3 quadrant domain
        _hypothesis(yaw_class="1"),        # a string must not be coerced
        _hypothesis(support=-1),           # negative support
        _hypothesis(spatial_bins=-2),      # negative spatial bins
        _hypothesis(passed="yes"),         # a non-boolean must not be truthy-coerced
        _hypothesis(passed=1),             # int 1 is not a boolean
        _hypothesis(refusal=None),         # a non-string must not be str-coerced
        # The four optional numeric diagnostics must never launder a malformed
        # value to None: a string, a boolean (plain and NumPy), a non-finite
        # value (plain and NumPy) and a non-scalar are all typed mismatches.
        _hypothesis(median_selected_residual_degrees="0.5"),
        _hypothesis(parallax_degrees=True),
        _hypothesis(cheirality_fraction=np.bool_(True)),
        _hypothesis(singular_ratio=float("nan")),
        _hypothesis(median_selected_residual_degrees=float("inf")),
        _hypothesis(parallax_degrees=-float("inf")),
        _hypothesis(cheirality_fraction=np.float64("nan")),
        _hypothesis(singular_ratio=[0.02]),
        None,                              # not a diagnostic object at all
    )
    for hypothesis in malformed:
        result = _refused_result("a", "b")
        result.hypotheses = (hypothesis,)
        with pytest.raises(ImageDerivedConnectionError) as info:
            build_image_derived_connections(
                session, budgets=_budgets(), matcher=_matcher_returning(result))
        assert info.value.reason == CONNECTIONS_MATCHER_MISMATCH

    # A non-sequence hypotheses field is refused just as typed.
    result = _refused_result("a", "b")
    result.hypotheses = 7
    with pytest.raises(ImageDerivedConnectionError) as info:
        build_image_derived_connections(
            session, budgets=_budgets(), matcher=_matcher_returning(result))
    assert info.value.reason == CONNECTIONS_MATCHER_MISMATCH


def test_a_malformed_top_level_matcher_diagnostic_is_a_typed_contract_error(monkeypatch):
    """A malformed top-level diagnostic must refuse, not become None evidence.

    ``_finite_or_none`` used to reduce a string, a boolean or a non-finite value
    to ``None``, so a non-conforming matcher could record fabricated
    "unmeasured" diagnostics instead of being refused. Every field must now
    raise the typed matcher-contract error, and the message must name it.
    """
    session = _simple_session(monkeypatch, ("a", "b"))
    fields = (
        "median_selected_residual_degrees",
        "parallax_degrees",
        "cheirality_fraction",
        "singular_ratio",
    )
    malformed = (
        "0.5", True, np.bool_(True), float("nan"), float("inf"), -float("inf"),
        np.float64("nan"), np.float32("inf"), [0.5], object(),
    )
    for name in fields:
        for value in malformed:
            result = _refused_result("a", "b")
            setattr(result, name, value)
            with pytest.raises(ImageDerivedConnectionError) as info:
                build_image_derived_connections(
                    session, budgets=_budgets(), matcher=_matcher_returning(result))
            assert info.value.reason == CONNECTIONS_MATCHER_MISMATCH
            assert name in info.value.detail

    # The same strictness holds on a solved result, whose other diagnostics are
    # real; a single malformed field must still refuse the whole pair.
    for name in fields:
        result = _solved_result("a", "b", support=3)
        setattr(result, name, "not a number")
        with pytest.raises(ImageDerivedConnectionError) as info:
            build_image_derived_connections(
                session, budgets=_budgets(), matcher=_matcher_returning(result))
        assert info.value.reason == CONNECTIONS_MATCHER_MISMATCH


def test_legitimate_none_and_real_finite_matcher_diagnostics_are_preserved(monkeypatch):
    """Strictness must not break the matcher's legitimate None or exact floats."""
    session = _simple_session(monkeypatch, ("a", "b"))

    # A refused result legitimately carries all four diagnostics as None.
    graph = build_image_derived_connections(
        session, budgets=_budgets(), matcher=_matcher_returning(_refused_result("a", "b")))
    record = graph.evidence[0]
    assert record.status == "refused"
    assert record.median_selected_residual_degrees is None
    assert record.parallax_degrees is None
    assert record.cheirality_fraction is None
    assert record.singular_ratio is None

    # A solved result preserves exact finite Python and NumPy scalars.
    solved = _solved_result("a", "b", support=3)
    solved.median_selected_residual_degrees = np.float32(0.25)
    solved.parallax_degrees = np.int64(4)
    solved.cheirality_fraction = 1.0
    solved.singular_ratio = np.float64(0.02)
    graph = build_image_derived_connections(
        session, budgets=_budgets(), matcher=_matcher_returning(solved))
    record = graph.evidence[0]
    assert record.status == "solved"
    assert record.median_selected_residual_degrees == float(np.float32(0.25))
    assert record.parallax_degrees == 4.0
    assert record.cheirality_fraction == 1.0
    assert record.singular_ratio == 0.02
    assert all(type(value) is float for value in (
        record.median_selected_residual_degrees, record.parallax_degrees,
        record.cheirality_fraction, record.singular_ratio))


def test_valid_matcher_hypotheses_preserve_none_and_finite_numbers(monkeypatch):
    """The strict parser accepts real finite values and real hypothesis None."""
    session = _simple_session(monkeypatch, ("a", "b"))
    result = _refused_result("a", "b")
    result.hypotheses = (
        _hypothesis(),
        _hypothesis(median_selected_residual_degrees=None, parallax_degrees=None,
                    cheirality_fraction=None, singular_ratio=None),
        _hypothesis(parallax_degrees=np.int64(7), singular_ratio=np.float32(0.5)),
    )

    graph = build_image_derived_connections(
        session, budgets=_budgets(), matcher=_matcher_returning(result))
    first, second, third = graph.evidence[0].hypotheses

    assert first.median_selected_residual_degrees == 0.5
    assert first.parallax_degrees == 4.0
    assert first.cheirality_fraction == 1.0
    assert first.singular_ratio == 0.02
    assert (second.median_selected_residual_degrees, second.parallax_degrees,
            second.cheirality_fraction, second.singular_ratio) == (None, None, None, None)
    assert third.parallax_degrees == 7.0
    assert third.singular_ratio == float(np.float32(0.5))


def test_the_graph_record_carries_no_pixels_or_scene_geometry(monkeypatch):
    session = _simple_session(monkeypatch, ("a", "b"))

    def matcher(left_id, left_image, right_id, right_image, *, left_camera, right_camera):
        return _solved_result(left_id, right_id, support=2)

    graph = build_image_derived_connections(session, budgets=_budgets(), matcher=matcher)
    record = graph.to_record()

    keys = _keys(record)
    assert not keys & {"image", "pixels", "array", "pixel_data"}
    assert not keys & {"position", "translation", "scale", "bounds", "block", "transform"}
    assert record["session_sha256"] == session.session_sha256
    assert record["scale_known"] is False
    assert record["evidence"][0]["evidence_sha256"] == graph.evidence[0].evidence_sha256
