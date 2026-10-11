"""Public-output controls for the corrected joint matcher (DS-14).

Every scene is built by projecting independent 3D points through the shipped
``camera.project_points``; features/descriptors are then supplied through the
public injectable seam :func:`match_feature_frames`, so no module internals and no
SIFT are needed for the geometric controls. The estimate convention matches
DS-11: the right view's true yaw is 95 degrees, its estimate reports 5 degrees
modulo 90, and the matcher must recover right class 1.

Controls: exact scene solves through the *public* DS-11 pose; a repeated-descriptor
mass plus a geometrically consistent unique minority still solves the correct
class/baseline; identity self-match, few-point support and two competing baselines
refuse; the graph cap/truncation is reported; the selected set is one-to-one and
bit-identical across runs; the descriptor-footprint boundary rule drops contour
features and excluded RGB cannot reach a descriptor; unequal explicit intrinsics
still solve.
"""
import math

import numpy as np
import pytest

from mc_photogrammetry.pipeline.contracts import (
    CameraEstimate,
    CameraIntrinsics,
    CameraPose,
)
from mc_photogrammetry.pipeline.correspondence import FrameFeatures
from mc_photogrammetry.pipeline.joint_matching import (
    JOINT_REFUSAL_REASONS,
    JointMatchSettings,
    match_and_solve_pair,
    match_feature_frames,
)
from mc_photogrammetry.reconstruction.camera import (
    CameraModel,
    camera_rotation_matrix,
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


def _camera(position, yaw, pitch, *, intrinsics=None):
    return CameraModel(position, yaw, pitch, eye_height=0.0, intrinsics=intrinsics)


def _estimate(camera, *, width=WIDTH, height=HEIGHT):
    """Calibration-style estimate: yaw reported modulo 90 degrees."""
    return CameraEstimate(
        intrinsics=intrinsics_for(camera, width, height),
        pose=CameraPose(
            position=None,
            yaw_degrees=float(camera.yaw_degrees % 90.0),
            pitch_degrees=float(camera.pitch_degrees),
            roll_degrees=float(camera.roll_degrees),
        ),
    )


def _visible_points(points, left_camera, right_camera, *, width=WIDTH, height=HEIGHT):
    left_pixels = project_points(points, left_camera, width=width, height=height)
    right_pixels = project_points(points, right_camera, width=width, height=height)
    left, right = [], []
    for left_pixel, right_pixel in zip(left_pixels, right_pixels):
        assert left_pixel is not None and right_pixel is not None
        left.append((left_pixel[0], left_pixel[1]))
        right.append((right_pixel[0], right_pixel[1]))
    return np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)


def _features(name, pixels, descriptors, *, width=WIDTH, height=HEIGHT):
    return FrameFeatures(
        frame_id=name,
        points=np.asarray(pixels, dtype=np.float64),
        descriptors=np.asarray(descriptors, dtype=np.float64),
        width=width,
        height=height,
        usable_pixels=width * height,
    )


def _scene():
    left_camera = _camera(LEFT_POSITION, LEFT_YAW, LEFT_PITCH)
    right_camera = _camera(RIGHT_POSITION, RIGHT_YAW, RIGHT_PITCH)
    left_pixels, right_pixels = _visible_points(_POINTS, left_camera, right_camera)
    descriptors = np.random.default_rng(3).normal(size=(len(left_pixels), 16))
    return (
        _features("a", left_pixels, descriptors),
        _features("b", right_pixels, descriptors),
        _estimate(left_camera),
        _estimate(right_camera),
        left_camera,
        right_camera,
    )


def _rotation_angle_degrees(first, second):
    product = np.asarray(first, dtype=np.float64).T @ np.asarray(second, dtype=np.float64)
    cosine = np.clip((np.trace(product) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def _direction_angle_degrees(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    return math.degrees(math.acos(float(np.clip(np.dot(first, second), -1.0, 1.0))))


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def _expected_baseline():
    return _unit(np.asarray(RIGHT_POSITION) - np.asarray(LEFT_POSITION))


def _one_to_one(matches):
    left = {(round(float(x), 6), round(float(y), 6)) for x, y in matches[:, 0:2]}
    right = {(round(float(x), 6), round(float(y), 6)) for x, y in matches[:, 2:4]}
    return len(left) == len(matches) == len(right)


def _ambiguous_scene():
    """A large ambiguous descriptor mass plus a consistent unique minority.

    The first ``n_repeated`` keypoints carry near-identical descriptors and are
    placed at independent random pixels on each side, so the descriptor graph
    confidently pairs points that have no consistent two-view geometry. Only the
    trailing unique minority is projected from real 3D points.
    """
    left_camera = _camera(LEFT_POSITION, LEFT_YAW, LEFT_PITCH)
    right_camera = _camera(RIGHT_POSITION, RIGHT_YAW, RIGHT_PITCH)
    unique_left, unique_right = _visible_points(_POINTS, left_camera, right_camera)
    generator = np.random.default_rng(11)
    n_repeated = 40
    repeated_left = np.column_stack(
        (generator.uniform(40, WIDTH - 40, n_repeated), generator.uniform(40, HEIGHT - 40, n_repeated))
    )
    repeated_right = np.column_stack(
        (generator.uniform(40, WIDTH - 40, n_repeated), generator.uniform(40, HEIGHT - 40, n_repeated))
    )
    jitter = generator.normal(scale=0.05, size=(n_repeated, 16))
    unique = generator.normal(size=(len(unique_left), 16))
    left = _features(
        "a",
        np.vstack((repeated_left, unique_left)),
        np.vstack((jitter, unique)),
    )
    right = _features(
        "b",
        np.vstack((repeated_right, unique_right)),
        np.vstack((jitter, unique)),
    )
    return left, right, _estimate(left_camera), _estimate(right_camera)


# --- exact geometry through the public DS-11 pose ---------------------------


def test_exact_scene_solves_through_public_ds11_pose():
    left, right, left_estimate, right_estimate, left_camera, right_camera = _scene()
    result = match_feature_frames(left, right, left_estimate, right_estimate)

    assert result.status == "solved", result.reason
    assert result.reason == ""
    assert result.solved is True
    assert result.right_yaw_class == RIGHT_YAW_CLASS
    assert result.pose is not None and result.pose.solved is True
    assert result.pose.right_yaw_class == result.right_yaw_class
    assert result.support >= result.support_floor >= 8
    assert min(result.spatial_bins_left, result.spatial_bins_right) >= 4
    assert result.median_selected_residual_degrees is not None
    assert result.median_selected_residual_degrees <= 1.0
    assert result.parallax_degrees is not None and result.parallax_degrees >= 3.0
    assert result.cheirality_fraction is not None and result.cheirality_fraction >= 0.75
    assert result.ambiguity_margin is not None and result.ambiguity_margin >= 2
    assert result.truncated == 0
    assert _one_to_one(result.matches)
    assert result.matches.shape[0] == result.support
    assert all(
        hypothesis.refusal in ("",) + JOINT_REFUSAL_REASONS
        for hypothesis in result.hypotheses
    )

    expected_rotation = camera_rotation_matrix(right_camera) @ camera_rotation_matrix(left_camera).T
    assert _rotation_angle_degrees(result.pose.relative_rotation, expected_rotation) <= 1.0
    assert _direction_angle_degrees(result.pose.baseline_direction, _expected_baseline()) <= 1.0


def test_selected_support_is_one_to_one_and_deterministic():
    left, right, left_estimate, right_estimate = _scene()[:4]
    first = match_feature_frames(left, right, left_estimate, right_estimate)
    second = match_feature_frames(left, right, left_estimate, right_estimate)

    assert first.solved is True and second.solved is True
    np.testing.assert_array_equal(first.matches, second.matches)
    assert first.right_yaw_class == second.right_yaw_class
    assert first.support == second.support
    assert [score.support for score in first.hypotheses] == [
        score.support for score in second.hypotheses
    ]
    assert _one_to_one(first.matches)


def test_orientation_rows_do_not_multiply_independent_geometric_support():
    left, right, left_estimate, right_estimate = _scene()[:4]
    copies = np.repeat(np.arange(len(left.points)), 3)
    duplicated_left = _features('a', left.points[copies], left.descriptors[copies])
    duplicated_right = _features('b', right.points[copies], right.descriptors[copies])
    result = match_feature_frames(duplicated_left, duplicated_right, left_estimate, right_estimate)
    assert result.solved, result.reason
    assert result.raw_keypoints == (60, 60)  # descriptor rows remain observable
    assert result.support == len(left.points) == 20
    assert _one_to_one(result.matches)
    assert result.right_yaw_class == RIGHT_YAW_CLASS
    assert _direction_angle_degrees(result.pose.baseline_direction, _expected_baseline()) <= 1.


def test_orientation_rows_cannot_fabricate_minimum_support():
    left, right, left_estimate, right_estimate = _scene()[:4]
    copies = np.repeat(np.arange(7), 3)
    result = match_feature_frames(
        _features('a', left.points[copies], left.descriptors[copies]),
        _features('b', right.points[copies], right.descriptors[copies]),
        left_estimate, right_estimate)
    assert not result.solved
    assert result.reason == 'insufficient_one_to_one_support'
    assert result.support < result.support_floor == 8


def test_duplicate_orientation_rows_cannot_break_physical_baseline_ambiguity():
    left_camera = _camera(LEFT_POSITION, LEFT_YAW, LEFT_PITCH)
    right_a = _camera(RIGHT_POSITION, RIGHT_YAW, RIGHT_PITCH)
    right_b = _camera((2.5, 3., -1.), RIGHT_YAW, RIGHT_PITCH)
    left_pixels, a_pixels = _visible_points(_POINTS, left_camera, right_a)
    _, b_pixels = _visible_points(_POINTS, left_camera, right_b)
    descriptors = np.random.default_rng(4).normal(size=(len(left_pixels), 16))
    # Twenty independent points support each distinct baseline. Two extra SIFT
    # orientation rows for A are the same observed geometry, not a margin of 2.
    extra = np.asarray((0, 1))
    left = _features('a', np.vstack((left_pixels, left_pixels, left_pixels[extra])),
                     np.vstack((descriptors, descriptors, descriptors[extra])))
    right = _features('b', np.vstack((a_pixels, b_pixels, a_pixels[extra])),
                      np.vstack((descriptors, descriptors, descriptors[extra])))
    result = match_feature_frames(left, right, _estimate(left_camera), _estimate(right_a))
    assert not result.solved
    assert result.reason == 'ambiguous_hypothesis', result.reason
    assert result.support == 20
    assert result.ambiguity_margin == 0
    assert _one_to_one(result.matches)


def test_pixel_endpoint_identity_is_exact_and_preserves_unique_row_order():
    from mc_photogrammetry.pipeline.joint_matching import _pixel_endpoint_ids
    adjacent = np.nextafter(4., np.inf)
    points = np.asarray(((4., 8.), (1., 2.), (4., 8.), (adjacent, 8.), (1., 2.)))
    np.testing.assert_array_equal(_pixel_endpoint_ids(points), (0, 1, 0, 3, 1))
    np.testing.assert_array_equal(_pixel_endpoint_ids(points[[3, 1, 0]]), (0, 1, 2))


@pytest.mark.parametrize('duplicate_side', ['left', 'right'])
def test_greedy_support_rejects_reused_pixel_on_either_view(duplicate_side):
    from mc_photogrammetry.pipeline.correspondence import ReciprocalGraph
    from mc_photogrammetry.pipeline.joint_matching import _greedy_selection
    graph = ReciprocalGraph(np.asarray(((0, 0), (1, 1), (2, 2))),
                            np.asarray((1., 2., 3.)), 3, 3, 3, 0)
    ids = np.asarray((0, 0, 2))
    independent = np.arange(3)
    endpoint_ids = (ids, independent) if duplicate_side == 'left' else (independent, ids)
    rows = np.asarray(((1., 0., 0.),) * 3)
    selection = _greedy_selection(rows, np.ones(3), graph, np.asarray((0., 0., 1.)),
                                  1., endpoint_ids=endpoint_ids)
    np.testing.assert_array_equal(selection, (True, False, True))


def test_unique_coordinate_matcher_retains_legacy_selection_exactly(monkeypatch):
    import mc_photogrammetry.pipeline.joint_matching as matcher_module
    left, right, left_estimate, right_estimate = _scene()[:4]
    current = match_feature_frames(left, right, left_estimate, right_estimate)
    monkeypatch.setattr(matcher_module, '_pixel_endpoint_ids', lambda points: np.arange(len(points)))
    legacy_identity = match_feature_frames(left, right, left_estimate, right_estimate)
    np.testing.assert_array_equal(current.matches, legacy_identity.matches)
    np.testing.assert_array_equal(current.pose.baseline_direction, legacy_identity.pose.baseline_direction)
    assert current.hypotheses == legacy_identity.hypotheses
    assert current.support == legacy_identity.support
    assert current.ambiguity_margin == legacy_identity.ambiguity_margin


def test_duplicate_rows_do_not_raise_physical_support_floor():
    left, right, left_estimate, right_estimate = _scene()[:4]
    settings = JointMatchSettings(min_support_fraction=.5, ransac_iterations=32)
    original = match_feature_frames(left, right, left_estimate, right_estimate, matcher=settings)
    duplicated = match_feature_frames(
        _features('a', np.repeat(left.points, 3, axis=0), np.repeat(left.descriptors, 3, axis=0)),
        _features('b', np.repeat(right.points, 3, axis=0), np.repeat(right.descriptors, 3, axis=0)),
        left_estimate, right_estimate, matcher=settings)
    assert original.support_floor == duplicated.support_floor == 10
    assert original.solved and duplicated.solved
    assert original.support == duplicated.support == 20
    assert original.raw_keypoints == (20, 20)
    assert duplicated.raw_keypoints == (60, 60)  # Diagnostic remains raw rows.


@pytest.mark.parametrize('replicas', [1, 5])
def test_duplicate_heavy_modes_cannot_displace_independent_shortlist(monkeypatch, replicas):
    import mc_photogrammetry.pipeline.joint_matching as matcher_module
    from mc_photogrammetry.pipeline.correspondence import ReciprocalGraph
    # Freeze only hypothesis generation, not support measurement. Three modes
    # have 3 independent endpoints each; the last two have 8 and 9. Duplicating the weak
    # rows makes each raw count 15, enough to steal all three shortlist slots.
    candidates = np.asarray(((1., 0., 0.), (0., 1., 0.), (0., 0., 1.),
                             (1., 1., 1.), (1., -1., 1.)))
    candidates /= np.linalg.norm(candidates, axis=1)[:, None]
    generator = np.random.default_rng(917)
    rows, identities = [], []
    identity = 0
    for index, candidate in enumerate(candidates):
        count = (8, 9)[index - 3] if index >= 3 else 3
        repeat = 1 if index >= 3 else replicas
        for _ in range(count):
            # Reject incidental support for another frozen direction.
            while True:
                row = generator.normal(size=3)
                row -= np.dot(row, candidate) * candidate
                row /= np.linalg.norm(row)
                others = np.delete(candidates, index, axis=0)
                if np.min(np.abs(others @ row)) > .05:
                    break
            rows.extend([row] * repeat)
            identities.extend([identity] * repeat)
            identity += 1
    rows = np.asarray(rows)
    graph = ReciprocalGraph(np.column_stack((np.arange(len(rows)), np.arange(len(rows)))),
                            np.zeros(len(rows)), len(rows), len(rows), len(rows), 0)
    endpoint_ids = (np.asarray(identities), np.asarray(identities))
    hypotheses = iter(candidates)
    monkeypatch.setattr(matcher_module, '_null_vector', lambda rows: next(hypotheses))
    monkeypatch.setattr(matcher_module, '_sample_rank_ok', lambda rows, threshold: True)

    def fixed_refit(rows, norms, graph, candidate, bounds, *, endpoint_ids=None):
        selection = matcher_module._greedy_selection(
            rows, norms, graph, candidate, bounds.inlier_residual_degrees,
            endpoint_ids=endpoint_ids)
        return None if selection is None else (candidate, selection)

    monkeypatch.setattr(matcher_module, '_select_and_refit', fixed_refit)
    modes = matcher_module._class_modes(rows, np.linalg.norm(rows, axis=1), graph,
        JointMatchSettings(ransac_iterations=4, max_modes_per_class=3), 8,
        endpoint_ids=endpoint_ids)
    assert len(modes) == 2
    np.testing.assert_allclose(modes[0][0], candidates[4])
    np.testing.assert_allclose(modes[1][0], candidates[3])
    assert [int(np.count_nonzero(mode[1])) for mode in modes] == [9, 8]
    from types import SimpleNamespace
    winner = SimpleNamespace(support=9, spatial_bins_left=4, spatial_bins_right=4,
                             singular_ratio=.1, parallax=5., cheirality=1.)
    assert matcher_module._winner_refusal(winner, [], 8, 1, JointMatchSettings()) == 'ambiguous_hypothesis'


def test_unequal_explicit_intrinsics_still_solve():
    left_intrinsics = CameraIntrinsics((1280, 720), (900.0, 880.0), (600.0, 380.0))
    right_intrinsics = CameraIntrinsics((1024, 768), (700.0, 760.0), (540.0, 350.0))
    left_camera = _camera(LEFT_POSITION, LEFT_YAW, LEFT_PITCH, intrinsics=left_intrinsics)
    right_camera = _camera(RIGHT_POSITION, RIGHT_YAW, RIGHT_PITCH, intrinsics=right_intrinsics)
    left_projected = project_points(_POINTS, left_camera, width=1280, height=720)
    right_projected = project_points(_POINTS, right_camera, width=1024, height=768)
    left_pixels = np.asarray([[p[0], p[1]] for p in left_projected], dtype=np.float64)
    right_pixels = np.asarray([[p[0], p[1]] for p in right_projected], dtype=np.float64)
    generator = np.random.default_rng(3)
    descriptors = generator.normal(size=(len(left_pixels), 16))
    result = match_feature_frames(
        _features("a", left_pixels, descriptors, width=1280, height=720),
        _features("b", right_pixels, descriptors, width=1024, height=768),
        CameraEstimate(left_intrinsics, CameraPose(None, LEFT_YAW % 90.0, LEFT_PITCH, 0.0)),
        CameraEstimate(right_intrinsics, CameraPose(None, RIGHT_YAW % 90.0, RIGHT_PITCH, 0.0)),
    )

    assert result.solved is True
    assert result.right_yaw_class == RIGHT_YAW_CLASS
    expected_rotation = camera_rotation_matrix(right_camera) @ camera_rotation_matrix(left_camera).T
    assert _rotation_angle_degrees(result.pose.relative_rotation, expected_rotation) <= 1.0
    assert _direction_angle_degrees(result.pose.baseline_direction, _expected_baseline()) <= 1.0


# --- repetition and ambiguity ------------------------------------------------


def test_repeated_descriptors_plus_unique_minority_solves_correct_class():
    left, right, left_estimate, right_estimate = _ambiguous_scene()
    result = match_feature_frames(left, right, left_estimate, right_estimate)

    assert result.candidate_edges > 0
    assert result.solved is True, result.reason
    assert result.right_yaw_class == RIGHT_YAW_CLASS
    assert result.support >= result.support_floor >= 8
    assert _one_to_one(result.matches)
    # The descriptor mass contributes a couple of chance inliers, so the solved
    # baseline is close but not sub-degree; class correctness is the claim.
    assert _direction_angle_degrees(result.pose.baseline_direction, _expected_baseline()) <= 3.0
    # The all-edge median is a diagnostic only: it may exceed the 1 degree band
    # while the selected support is exactly consistent.
    winner = [h for h in result.hypotheses if h.yaw_class == RIGHT_YAW_CLASS][0]
    assert winner.median_selected_residual_degrees <= 1.0
    assert result.median_selected_residual_degrees is not None


def test_two_competing_baselines_refuse_ambiguous():
    left_camera = _camera(LEFT_POSITION, LEFT_YAW, LEFT_PITCH)
    right_a = _camera(RIGHT_POSITION, RIGHT_YAW, RIGHT_PITCH)
    right_b = _camera((2.5, 3.0, -1.0), RIGHT_YAW, RIGHT_PITCH)
    left_a, a_pixels = _visible_points(_POINTS, left_camera, right_a)
    left_b, b_pixels = _visible_points(_POINTS, left_camera, right_b)
    count = min(len(left_a), len(left_b))
    descriptors = np.random.default_rng(4).normal(size=(count, 16))
    left = _features(
        "a", np.vstack((left_a[:count], left_b[:count])), np.vstack((descriptors, descriptors))
    )
    right = _features(
        "b", np.vstack((a_pixels[:count], b_pixels[:count])), np.vstack((descriptors, descriptors))
    )
    result = match_feature_frames(left, right, _estimate(left_camera), _estimate(right_a))

    assert result.solved is False
    assert result.reason == "ambiguous_hypothesis", result.reason
    assert result.pose is None
    assert result.right_yaw_class is None
    assert result.support >= result.support_floor
    assert result.ambiguity_margin is not None and result.ambiguity_margin < 2


def test_identity_self_match_refuses():
    left, _right, left_estimate = _scene()[:3]
    right = _features("b", left.points.copy(), left.descriptors.copy())
    result = match_feature_frames(left, right, left_estimate, left_estimate)

    assert result.solved is False
    assert result.reason in JOINT_REFUSAL_REASONS
    assert result.pose is None
    assert result.right_yaw_class is None


def test_insufficient_support_refuses_with_typed_reason():
    left, right, left_estimate, right_estimate = _scene()[:4]
    few_left = _features("a", left.points[:5], left.descriptors[:5])
    few_right = _features("b", right.points[:5], right.descriptors[:5])
    result = match_feature_frames(few_left, few_right, left_estimate, right_estimate)

    assert result.solved is False
    assert result.reason in JOINT_REFUSAL_REASONS
    assert result.pose is None
    assert result.support < result.support_floor


def test_no_mode_refusal_reports_the_computed_support_floor():
    """A refusal with no qualifying mode still reports the computed floor.

    Five-point frames cap one-to-one support at five, so a floor of twenty leaves
    every candidate mode below it and the matcher refuses before ranking any mode.
    That refusal must carry the computed floor rather than the ``_refused`` default
    of zero.
    """
    left, right, left_estimate, right_estimate = _scene()[:4]
    few_left = _features("a", left.points[:5], left.descriptors[:5])
    few_right = _features("b", right.points[:5], right.descriptors[:5])
    settings = JointMatchSettings(min_support_absolute=20)
    expected_floor = max(
        int(settings.min_support_absolute),
        int(math.ceil(settings.min_support_fraction * min(len(few_left.points), len(few_right.points)))),
    )

    result = match_feature_frames(
        few_left, few_right, left_estimate, right_estimate, matcher=settings
    )

    assert result.solved is False
    assert result.pose is None
    assert result.reason == "insufficient_one_to_one_support", result.reason
    assert result.support_floor == expected_floor == 20
    assert result.support < result.support_floor


def test_edge_cap_truncates_and_is_reported():
    left, right, left_estimate, right_estimate = _ambiguous_scene()
    capped = JointMatchSettings(max_candidate_edges=8)
    result = match_feature_frames(left, right, left_estimate, right_estimate, matcher=capped)
    uncapped = match_feature_frames(left, right, left_estimate, right_estimate)

    assert result.candidate_edges == 8
    assert result.total_graph_edges == uncapped.total_graph_edges
    assert result.total_graph_edges > 8
    assert result.truncated == result.total_graph_edges - 8


# --- descriptor-footprint boundary and excluded RGB --------------------------


def _noise_frame(generator, height, width):
    return generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def test_boundary_footprint_filter_and_excluded_rgb_invariance():
    height, width = 240, 320
    usable = np.zeros((height, width), dtype=bool)
    usable[40:200, 40:280] = True
    masks = {"left.png": usable, "right.png": usable}
    generator = np.random.default_rng(9)
    left = _noise_frame(generator, height, width)
    right = _noise_frame(generator, height, width)
    camera = _camera((0.0, 3.0, 0.0), LEFT_YAW, LEFT_PITCH)
    estimate = _estimate(camera, width=width, height=height)
    other = _estimate(_camera(RIGHT_POSITION, RIGHT_YAW, RIGHT_PITCH), width=width, height=height)

    baseline = match_and_solve_pair(
        "left.png", left, "right.png", right,
        left_camera=estimate, right_camera=other, usable_masks=masks,
    )
    assert baseline.boundary_dropped[0] > 0 and baseline.boundary_dropped[1] > 0

    mutated_left = left.copy()
    mutated_right = right.copy()
    mutated_left[~usable] = 255
    mutated_right[~usable] = 0
    changed = match_and_solve_pair(
        "left.png", mutated_left, "right.png", mutated_right,
        left_camera=estimate, right_camera=other, usable_masks=masks,
    )

    np.testing.assert_array_equal(baseline.matches, changed.matches)
    assert baseline.candidate_edges == changed.candidate_edges
    assert baseline.boundary_dropped == changed.boundary_dropped
    assert baseline.raw_keypoints == changed.raw_keypoints
    assert (baseline.status, baseline.reason) == (changed.status, changed.reason)


def test_default_detection_keeps_frozen_ds12_boundary_behaviour():
    left, right, left_estimate, right_estimate = _scene()[:4]
    result = match_feature_frames(left, right, left_estimate, right_estimate)
    assert result.boundary_dropped == (0, 0)
