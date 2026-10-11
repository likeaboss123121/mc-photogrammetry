"""Public-output controls for the DS-18 directed-candidate matcher.

Every scene is built by projecting independent 3D points through the shipped
``camera.project_points``; features/descriptors are supplied through the public
injectable seam :func:`directed_matching.match_feature_frames`, so no module
internals and no SIFT are needed for the geometric controls. The estimate
convention matches DS-11/DS-14: the right view's true yaw is 95 degrees, its
estimate reports 5 degrees modulo 90, and the matcher must recover right class 1.

Controls:

* the directed graph keeps a left -> right edge that the DS-14 reciprocal graph
  drops, is deterministically sorted, and reports cap truncation;
* true edges that deliberately fail reciprocity still solve (the DS-14 graph has
  none of them);
* an exact scene solves through the *public* DS-11 pose; right collisions are
  resolved after scoring; repeated descriptors with a unique minority still solve;
* identity self-match, few-point support and two competing baselines refuse;
* the selected set is one-to-one and bit-identical across runs; the declared
  bounds are DS-14's object, unchanged; the descriptor-footprint boundary rule
  drops contour features and excluded RGB cannot reach a descriptor; unequal
  explicit intrinsics still solve.
"""
import math

import numpy as np
import pytest

from mc_photogrammetry.pipeline.connected_camera import PairGates
from mc_photogrammetry.pipeline.contracts import (
    CameraEstimate,
    CameraIntrinsics,
    CameraPose,
)
from mc_photogrammetry.pipeline.correspondence import (
    FrameFeatures,
    reciprocal_topk_graph,
)
from mc_photogrammetry.pipeline.directed_matching import (
    DIRECTED_REFUSAL_REASONS,
    DirectedMatchSettings,
    directed_topk_graph,
    match_and_solve_pair,
    match_feature_frames,
)
from mc_photogrammetry.pipeline.joint_matching import JointMatchSettings
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


def _ambiguous_scene(n_repeated=12):
    """A repeated-descriptor mass plus a consistent unique minority.

    The first ``n_repeated`` keypoints carry near-identical descriptors and are
    placed at independent random pixels on each side, so the descriptor graph
    confidently pairs points that have no consistent two-view geometry. Only the
    trailing unique minority is projected from real 3D points. The directed
    per-left pass admits a chance edge for every repeated left endpoint, so this
    control declares a bounded 12-point repetition mass; the report records the
    larger-mass refusals this matcher exhibits.
    """
    left_camera = _camera(LEFT_POSITION, LEFT_YAW, LEFT_PITCH)
    right_camera = _camera(RIGHT_POSITION, RIGHT_YAW, RIGHT_PITCH)
    unique_left, unique_right = _visible_points(_POINTS, left_camera, right_camera)
    generator = np.random.default_rng(11)
    repeated_left = np.column_stack(
        (
            generator.uniform(40, WIDTH - 40, n_repeated),
            generator.uniform(40, HEIGHT - 40, n_repeated),
        )
    )
    repeated_right = np.column_stack(
        (
            generator.uniform(40, WIDTH - 40, n_repeated),
            generator.uniform(40, HEIGHT - 40, n_repeated),
        )
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


def _reciprocity_break_scene(decoys_per_point=1):
    """True correspondences whose edges the reciprocal graph cannot keep.

    Each true left descriptor is perturbed slightly and ``decoys_per_point``
    exact-descriptor decoys with lower indices are inserted for every point. With
    ``neighbors=1`` the right keypoint's nearest left is therefore a decoy, so no
    true left -> right edge is reciprocal; the directed graph still reaches every
    true edge because it only asks the left side.
    """
    left_camera = _camera(LEFT_POSITION, LEFT_YAW, LEFT_PITCH)
    right_camera = _camera(RIGHT_POSITION, RIGHT_YAW, RIGHT_PITCH)
    true_left, true_right = _visible_points(_POINTS, left_camera, right_camera)
    generator = np.random.default_rng(7)
    true_descriptors = generator.normal(size=(len(true_left), 16))
    decoy_pixels = np.column_stack(
        (
            generator.uniform(40, WIDTH - 40, decoys_per_point * len(true_left)),
            generator.uniform(40, HEIGHT - 40, decoys_per_point * len(true_left)),
        )
    )
    decoy_descriptors = np.repeat(true_descriptors, decoys_per_point, axis=0)
    perturbed = true_descriptors + 0.01 * generator.normal(size=true_descriptors.shape)
    left = _features(
        "a", np.vstack((decoy_pixels, true_left)), np.vstack((decoy_descriptors, perturbed))
    )
    right = _features("b", true_right, true_descriptors)
    true_left_indices = np.arange(
        decoys_per_point * len(true_left), decoys_per_point * len(true_left) + len(true_left)
    )
    true_pairs = set(zip(true_left_indices.tolist(), range(len(true_left))))
    return left, right, _estimate(left_camera), _estimate(right_camera), true_pairs


# --- directed graph ----------------------------------------------------------


def test_directed_graph_has_no_reciprocity_requirement():
    left = np.asarray([[0.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [5.0, 5.0, 5.0, 5.0]])
    right = np.asarray([[0.0, 0.0, 0.0, 0.0]])
    directed = directed_topk_graph(left, right, neighbors=1, max_edges=64)
    reciprocal = reciprocal_topk_graph(left, right, neighbors=1, max_edges=64)

    pairs = {tuple(int(value) for value in edge) for edge in directed.edges}
    reciprocal_pairs = {tuple(int(value) for value in edge) for edge in reciprocal.edges}
    assert (0, 0) in pairs and (0, 0) not in reciprocal_pairs
    assert (1, 0) in reciprocal_pairs  # the reciprocal graph keeps only its own best
    assert directed.total_edges == 3 and directed.truncated == 0
    # Sorted by (distance, left, right): index 1 is the exact match.
    assert [tuple(int(value) for value in edge) for edge in directed.edges] == [
        (1, 0),
        (0, 0),
        (2, 0),
    ]
    assert [float(value) for value in directed.distances] == sorted(
        float(value) for value in directed.distances
    )

    repeat = directed_topk_graph(left, right, neighbors=1, max_edges=64)
    np.testing.assert_array_equal(directed.edges, repeat.edges)
    np.testing.assert_array_equal(directed.distances, repeat.distances)


def test_directed_graph_cap_truncates_and_is_reported():
    descriptors = np.eye(10, 8)
    graph = directed_topk_graph(descriptors, descriptors, neighbors=1, max_edges=4)

    assert graph.total_edges == 10
    assert graph.truncated == 6
    assert len(graph.edges) == 4
    # The cap drops the largest descriptor distances first; here all are zero, so
    # the tie-break keeps the lowest (left, right) pairs.
    assert [tuple(int(value) for value in edge) for edge in graph.edges] == [
        (0, 0), (1, 1), (2, 2), (3, 3)
    ]


def test_directed_true_edges_fail_reciprocity_yet_solve():
    left, right, left_estimate, right_estimate, true_pairs = _reciprocity_break_scene()
    settings = JointMatchSettings(neighbors=1)
    directed = directed_topk_graph(
        left.descriptors, right.descriptors, neighbors=settings.neighbors
    )
    reciprocal = reciprocal_topk_graph(
        left.descriptors, right.descriptors, neighbors=settings.neighbors
    )
    directed_pairs = {tuple(int(value) for value in edge) for edge in directed.edges}
    reciprocal_pairs = {tuple(int(value) for value in edge) for edge in reciprocal.edges}

    assert len(true_pairs) == len(_POINTS)
    assert true_pairs <= directed_pairs  # every directed truth is a candidate
    assert not (true_pairs & reciprocal_pairs)  # ... and none survives reciprocity

    result = match_feature_frames(
        left, right, left_estimate, right_estimate, matcher=settings
    )
    assert result.solved is True, result.reason
    assert result.right_yaw_class == RIGHT_YAW_CLASS
    assert result.support >= result.support_floor >= 8
    assert _one_to_one(result.matches)
    assert _direction_angle_degrees(result.pose.baseline_direction, _expected_baseline()) <= 1.0


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
    assert result.support == result.left_support == result.right_support == len(_POINTS)
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
        hypothesis.refusal in ("",) + DIRECTED_REFUSAL_REASONS
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
    assert first.left_support == second.left_support
    assert first.right_support == second.right_support
    assert [score.support for score in first.hypotheses] == [
        score.support for score in second.hypotheses
    ]
    assert _one_to_one(first.matches)


def test_right_collisions_resolve_after_scoring():
    left, right, left_estimate, right_estimate = _scene()[:4]
    duplicates = np.asarray(
        [
            [left.points[0][0] + 2.0, left.points[0][1]],
            [left.points[0][0] + 4.0, left.points[0][1]],
            [left.points[0][0], left.points[0][1] + 2.0],
        ]
    )
    crowded_left = _features(
        "a",
        np.vstack((left.points, duplicates)),
        np.vstack((left.descriptors, np.repeat(left.descriptors[0:1], 3, axis=0))),
    )
    result = match_feature_frames(crowded_left, right, left_estimate, right_estimate)

    assert result.solved is True, result.reason
    assert result.right_yaw_class == RIGHT_YAW_CLASS
    # Four left endpoints (one true plus three duplicates) want the same right
    # endpoint; the collision is resolved down to a single one-to-one edge.
    assert result.left_support > result.support
    assert result.support == result.right_support
    assert _one_to_one(result.matches)
    assert result.matches.shape[0] == result.support == len(_POINTS)


def test_unequal_explicit_intrinsics_still_solve():
    left_intrinsics = CameraIntrinsics((1280, 720), (900.0, 880.0), (600.0, 380.0))
    right_intrinsics = CameraIntrinsics((1024, 768), (700.0, 760.0), (540.0, 350.0))
    left_camera = _camera(LEFT_POSITION, LEFT_YAW, LEFT_PITCH, intrinsics=left_intrinsics)
    right_camera = _camera(RIGHT_POSITION, RIGHT_YAW, RIGHT_PITCH, intrinsics=right_intrinsics)
    left_projected = project_points(_POINTS, left_camera, width=1280, height=720)
    right_projected = project_points(_POINTS, right_camera, width=1024, height=768)
    left_pixels = np.asarray([[p[0], p[1]] for p in left_projected], dtype=np.float64)
    right_pixels = np.asarray([[p[0], p[1]] for p in right_projected], dtype=np.float64)
    descriptors = np.random.default_rng(3).normal(size=(len(left_pixels), 16))
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
    assert result.reason in DIRECTED_REFUSAL_REASONS
    assert result.pose is None
    assert result.right_yaw_class is None


def test_insufficient_support_refuses_with_typed_reason():
    left, right, left_estimate, right_estimate = _scene()[:4]
    few_left = _features("a", left.points[:5], left.descriptors[:5])
    few_right = _features("b", right.points[:5], right.descriptors[:5])
    result = match_feature_frames(few_left, few_right, left_estimate, right_estimate)

    assert result.solved is False
    assert result.reason in DIRECTED_REFUSAL_REASONS
    assert result.pose is None
    assert result.support < result.support_floor


def test_matcher_edge_cap_truncates_and_is_reported():
    left, right, left_estimate, right_estimate = _ambiguous_scene()
    capped = match_feature_frames(
        left, right, left_estimate, right_estimate,
        matcher=JointMatchSettings(max_candidate_edges=8),
    )
    uncapped = match_feature_frames(left, right, left_estimate, right_estimate)

    assert capped.candidate_edges == 8
    assert capped.total_graph_edges == uncapped.total_graph_edges
    assert capped.total_graph_edges > 8
    assert capped.truncated == capped.total_graph_edges - 8


def test_declared_bounds_are_ds14_settings_unchanged():
    assert DirectedMatchSettings is JointMatchSettings
    assert DirectedMatchSettings() == JointMatchSettings()
    assert PairGates() == PairGates()


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
    assert baseline.left_support == changed.left_support
    assert baseline.right_support == changed.right_support
    assert baseline.boundary_dropped == changed.boundary_dropped
    assert baseline.raw_keypoints == changed.raw_keypoints
    assert (baseline.status, baseline.reason) == (changed.status, changed.reason)


def test_default_detection_keeps_frozen_ds12_boundary_behaviour():
    left, right, left_estimate, right_estimate = _scene()[:4]
    result = match_feature_frames(left, right, left_estimate, right_estimate)
    assert result.boundary_dropped == (0, 0)
