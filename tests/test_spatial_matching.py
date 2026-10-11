"""Spatial candidate controls: actual SIFT pixels, no changed pose gates."""
from dataclasses import fields

import numpy as np
import pytest

from mc_photogrammetry.pipeline.correspondence import CorrespondenceSettings, detect_features
from mc_photogrammetry.pipeline.joint_matching import JointMatchSettings, match_and_solve_pair, match_feature_frames
from mc_photogrammetry.pipeline.spatial_matching import _balanced_indices, detect_spatial_features
from mc_photogrammetry.pipeline.contracts import CameraEstimate, CameraPose
from mc_photogrammetry.reconstruction.camera import CameraModel, intrinsics_for


def texture():
    import cv2
    image = np.random.default_rng(714).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    return cv2.GaussianBlur(image, (3, 3), .7)


def assert_results_equal(first, second):
    for field in fields(first):
        left, right = getattr(first, field.name), getattr(second, field.name)
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif field.name == 'pose' and left is not None:
            for part in fields(left):
                a, b = getattr(left, part.name), getattr(right, part.name)
                if isinstance(a, np.ndarray):
                    np.testing.assert_array_equal(a, b)
                else:
                    assert a == b
        else:
            assert left == right


def test_cached_baseline_exact_pixel_entry_parity():
    image = texture()
    right = np.roll(image, 13, axis=1)
    mask = np.ones(image.shape[:2], bool)
    masks = {'a': mask, 'b': mask}
    bounds = JointMatchSettings(max_features=96, ransac_iterations=48)
    detection = CorrespondenceSettings(max_features=96, mask_boundary_pad_px=8.)
    camera = CameraModel((0., 0., 0.), 0., 0., eye_height=0.)
    estimate = CameraEstimate(intrinsics=intrinsics_for(camera, 320, 240),
                              pose=CameraPose(position=None, yaw_degrees=0., pitch_degrees=0., roll_degrees=0.))
    uncached = match_and_solve_pair('a', image, 'b', right, left_camera=estimate,
                                  right_camera=estimate, usable_masks=masks, matcher=bounds)
    left_features = detect_features('a', image, usable_mask=mask, settings=detection)
    right_features = detect_features('b', right, usable_mask=mask, settings=detection)
    cached = match_feature_frames(left_features, right_features, estimate, estimate, matcher=bounds)
    assert_results_equal(uncached, cached)


def test_strict_cell_quota_orientation_rows_and_determinism():
    points = []
    responses = []
    for cell in range(16):
        for index in range(300):
            points.append((cell % 4 * 100 + index % 80 + .2, cell // 4 * 100 + index // 80 + .2))
            responses.append(1000. if cell == 0 else 1. / (index + 1))
    points += [points[0]] * 20
    responses += [2000.] * 20
    points, responses = np.asarray(points), np.asarray(responses)
    chosen = _balanced_indices(points, responses, width=400, height=400, limit=4000)
    assert len(chosen) == 4000
    # Keep SIFT orientation rows: deduplication would be a second A/B variable.
    assert len(set(map(tuple, points[chosen]))) < 4000
    cells = (points[chosen, 0] // 100).astype(int) + 4 * (points[chosen, 1] // 100).astype(int)
    assert np.bincount(cells, minlength=16).tolist() == [250] * 16
    np.testing.assert_array_equal(chosen, _balanced_indices(points, responses, width=400, height=400, limit=4000))


def test_real_sift_excluded_rgb_invariance_and_quota():
    image = texture()
    mask = np.ones(image.shape[:2], bool)
    mask[65:175, 100:220] = False
    changed = image.copy()
    changed[~mask] = 255 - changed[~mask]
    settings = CorrespondenceSettings(max_features=160, mask_boundary_pad_px=8.)
    first = detect_spatial_features('a', image, usable_mask=mask, settings=settings)
    second = detect_spatial_features('a', changed, usable_mask=mask, settings=settings)
    assert len(first.points) > 16
    assert len(first.points) <= 160
    np.testing.assert_array_equal(first.points, second.points)
    np.testing.assert_array_equal(first.descriptors, second.descriptors)
    bins = (first.points[:, 0] * 4 / 320).astype(int) + 4 * (first.points[:, 1] * 4 / 240).astype(int)
    assert np.bincount(bins, minlength=16).max() <= 10


def test_same_pass_stage_census_distinguishes_filter_quota_and_descriptor_loss(monkeypatch):
    import cv2
    from types import SimpleNamespace
    raw = [SimpleNamespace(pt=(40. + index / 20, 40.), size=4., response=1.) for index in range(301)]
    raw += [SimpleNamespace(pt=(40. + index, 5.), size=4., response=1.) for index in range(30)]
    calls = []
    class Detector:
        def detect(self, grey, mask):
            calls.append('detect')
            assert not grey[:20].any()
            return raw
        def compute(self, grey, selected):
            calls.append(('compute', len(selected)))
            return selected[:-10], np.zeros((len(selected) - 10, 128), np.float32)
    monkeypatch.setattr(cv2, 'SIFT_create', lambda **kwargs: Detector())
    image = np.full((400, 400, 3), 100, np.uint8)
    mask = np.ones((400, 400), bool)
    mask[:20] = False
    census = {}
    feature = detect_spatial_features('a', image, usable_mask=mask,
        settings=CorrespondenceSettings(mask_boundary_pad_px=8.), _census=census)
    stages = ('raw_sift_proposals', 'footprint_retained', 'quota_selected', 'descriptor_returned')
    assert [census[stage]['count'] for stage in stages] == [331, 301, 250, 240]
    assert [sum(census[stage]['cells']) for stage in stages] == [331, 301, 250, 240]
    assert len(feature.points) == len(feature.descriptors) == 240
    assert calls == ['detect', ('compute', 250)]


@pytest.mark.parametrize('maximum', [0, True, 4001])
def test_candidate_rejects_bad_budget(maximum):
    with pytest.raises(ValueError):
        detect_spatial_features('a', texture(), settings=CorrespondenceSettings(max_features=maximum))


def test_missing_mask_is_authoritative_empty():
    from mc_photogrammetry.pipeline.spatial_matching import match_and_solve_spatial_pair
    image = texture()
    # Invalid cameras cannot create features from missing authoritative masks.
    result = match_and_solve_spatial_pair('a', image, 'b', image,
        left_camera=None, right_camera=None, usable_masks={})
    assert result.raw_keypoints == (0, 0)
    assert result.status == 'refused'


def _projected_texture_pair(background_sigma=24):
    """Known planar raster; one strong off-center patch starves global SIFT."""
    import cv2
    from mc_photogrammetry.reconstruction.camera import project_points
    random = np.random.default_rng(419)
    pixels = np.clip(128 + random.normal(0, background_sigma, (480, 640, 3)), 0, 255).astype(np.uint8)
    pixels[135:220, 170:300] = random.integers(0, 256, (85, 130, 3), dtype=np.uint8)
    pixels = cv2.GaussianBlur(pixels, (3, 3), .6)
    left_camera = CameraModel((0., 0., 0.), 4., 2., fov_degrees=75., eye_height=0.)
    right_camera = CameraModel((1.5, .2, .1), 10., -1., fov_degrees=75., eye_height=0.)
    world = [(-10., 7., 12.), (10., 7., 12.), (10., -7., 12.), (-10., -7., 12.)]
    corners = np.asarray([(0., 0.), (639., 0.), (639., 479.), (0., 479.)], np.float32)
    images, estimates = [], []
    for camera in (left_camera, right_camera):
        projected = np.asarray([point[:2] for point in project_points(world, camera, width=640, height=480)], np.float32)
        transform = cv2.getPerspectiveTransform(corners, projected)
        images.append(cv2.warpPerspective(pixels, transform, (640, 480)))
        estimates.append(CameraEstimate(intrinsics=intrinsics_for(camera, 640, 480),
            pose=CameraPose(position=None, yaw_degrees=camera.yaw_degrees,
                           pitch_degrees=camera.pitch_degrees, roll_degrees=0.)))
    return images, estimates


def test_spatial_candidate_planar_historical_false_accept_now_refuses_ambiguity():
    """Keep the raster fixed while checking the corrected physical shortlist.

    The original raw-row shortlist accepted a wrong pose (48.9677 degrees).
    Its exact controlled replay remains in the evaluator test; current matching
    must retain the competing physical mode and refuse, not claim a correct pose.
    """
    from mc_photogrammetry.pipeline.spatial_matching import match_and_solve_spatial_pair
    images, estimates = _projected_texture_pair()
    masks = {'a': np.ones((480, 640), bool), 'b': np.ones((480, 640), bool)}
    settings = JointMatchSettings()
    baseline = match_and_solve_pair('a', images[0], 'b', images[1],
        left_camera=estimates[0], right_camera=estimates[1], usable_masks=masks, matcher=settings)
    candidate = match_and_solve_spatial_pair('a', images[0], 'b', images[1],
        left_camera=estimates[0], right_camera=estimates[1], usable_masks=masks, matcher=settings)
    assert not candidate.solved
    assert candidate.reason == 'ambiguous_hypothesis'
    assert candidate.pose is None
    assert min(candidate.spatial_bins_left, candidate.spatial_bins_right) >= settings.min_spatial_bins
    assert candidate.ambiguity_margin < settings.ambiguity_support_margin


def test_spatial_candidate_distributed_texture_correct_pose():
    from mc_photogrammetry.pipeline.spatial_matching import match_and_solve_spatial_pair
    images, estimates = _projected_texture_pair(background_sigma=80)
    candidate = match_and_solve_spatial_pair('a', images[0], 'b', images[1],
        left_camera=estimates[0], right_camera=estimates[1],
        usable_masks={'a': np.ones((480, 640), bool), 'b': np.ones((480, 640), bool)})
    assert candidate.solved, candidate.reason
    expected = np.asarray((1.5, .2, .1))
    expected /= np.linalg.norm(expected)
    assert candidate.pose.right_yaw_class == 0
    assert np.degrees(np.arccos(np.clip(np.dot(expected, candidate.pose.baseline_direction), -1., 1.))) < 2.


@pytest.mark.parametrize('kind', ['self', 'independent'])
def test_spatial_candidate_pixel_negative_controls(kind):
    from mc_photogrammetry.pipeline.spatial_matching import match_and_solve_spatial_pair
    images, estimates = _projected_texture_pair()
    if kind == 'self':
        images[1], estimates[1] = images[0], estimates[0]
    else:
        images[1] = np.random.default_rng(882).integers(0, 256, images[0].shape, dtype=np.uint8)
    result = match_and_solve_spatial_pair('a', images[0], 'b', images[1],
        left_camera=estimates[0], right_camera=estimates[1],
        usable_masks={'a': np.ones((480, 640), bool), 'b': np.ones((480, 640), bool)},
        matcher=JointMatchSettings(max_features=128))
    assert not result.solved
