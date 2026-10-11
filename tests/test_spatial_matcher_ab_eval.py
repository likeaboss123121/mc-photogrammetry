"""Offline source/caching/truth-separation checks for the new product A/B."""
import importlib.util
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def test_direct_script_help_bootstraps_checkout_without_pytest_path(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/evaluation/evaluate_spatial_matcher_ab.py'
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    environment.pop('PYTHONHOME', None)
    # Isolated interpreter and unrelated cwd prevent pytest's src path or the
    # repository working directory from rescuing a missing direct-CLI bootstrap.
    result = subprocess.run([sys.executable, '-I', '-B', str(script), '--help'],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    for option in ('--input-manifest', '--evaluation-capture', '--evaluation-ground-truth', '--output'):
        assert option in result.stdout
    assert 'Traceback' not in result.stderr
    # Help exits during argument parsing: no image/metadata arguments supplied
    # and no outputs created in the unrelated working directory.
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/evaluation/evaluate_spatial_matcher_ab.py'
    spec = importlib.util.spec_from_file_location('spatial_matcher_ab_test', path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_cached_features_once_per_frame_downstream_default_gates(module, monkeypatch):
    from mc_photogrammetry.pipeline.correspondence import FrameFeatures
    from mc_photogrammetry.pipeline.joint_matching import JointMatchSettings, match_feature_frames
    from mc_photogrammetry.pipeline.connected_camera import PairGates
    frames = {f'image-{index:04d}': np.zeros((64, 96, 3), np.uint8) for index in range(6)}
    monkeypatch.setattr(module, 'build_image_derived_session', lambda frames: SimpleNamespace(
        cameras={name: None for name in frames}, session_sha256='s' * 64))
    cache, calls = {}, []
    def detector(arm):
        def run(frame_id, image, **kwargs):
            assert kwargs['settings'].max_features == 4000
            assert kwargs['settings'].mask_boundary_pad_px == 8.
            assert kwargs['usable_mask'].dtype == bool
            feature = FrameFeatures(frame_id, np.empty((0, 2)), np.empty((0, 0), np.float32),
                                    96, 64, int(kwargs['usable_mask'].sum()))
            assert (arm, frame_id) not in cache
            cache[arm, frame_id] = feature
            if '_census' in kwargs:
                kwargs['_census'].update({stage: module._stage_census(feature.points, 96, 64) for stage in
                    ('raw_sift_proposals', 'footprint_retained', 'quota_selected', 'descriptor_returned')})
            return feature
        return run
    baseline_detector = detector('baseline')
    def baseline_with_census(*args, **kwargs):
        feature = baseline_detector(*args, **kwargs)
        return feature, {stage: module._stage_census(feature.points, 96, 64) for stage in
            ('raw_sift_proposals', 'footprint_retained', 'quota_selected', 'descriptor_returned')}
    monkeypatch.setattr(module, '_detect_baseline_with_census', baseline_with_census)
    monkeypatch.setattr(module, 'detect_spatial_features', detector('spatial'))
    def match(left, right, left_camera, right_camera, **kwargs):
        assert kwargs['matcher'] == JointMatchSettings()
        assert kwargs['gates'] == PairGates()
        assert any(value is left for value in cache.values())
        assert any(value is right for value in cache.values())
        calls.append((left.frame_id, right.frame_id))
        return match_feature_frames(left, right, left_camera, right_camera)
    monkeypatch.setattr(module, 'match_feature_frames', match)
    # Pixel phase must not read truth or any file.
    monkeypatch.setattr(Path, 'read_text', lambda *a, **k: pytest.fail('truth/file read during pixel matching'))
    report, results, session = module.run_pixel_ab(frames)
    assert len(cache) == 12
    assert len(calls) == 30
    assert len(report['pairs']) == 30
    assert report['session_sha256'] == session.session_sha256
    assert set(results) == {'baseline', 'spatial'}
    assert all(row['image_camera']['session_sha256'] == session.session_sha256 for row in report['frames'])
    assert all(row['image_camera']['status'] == 'unavailable' for row in report['frames'])


@pytest.mark.parametrize('count', [0, 1, 7])
def test_frame_cap_before_calibration(module, monkeypatch, count):
    monkeypatch.setattr(module, 'build_image_derived_session', lambda *_: pytest.fail('calibration before cap'))
    with pytest.raises(ValueError):
        module.run_pixel_ab({str(index): np.zeros((2, 2, 3), np.uint8) for index in range(count)})


def test_topological_components_includes_unconnected_frames(module):
    assert module.components(['a', 'b', 'c'], [('a', 'b')]) == [['a', 'b'], ['c']]
    assert module.components(['a', 'b', 'c'], [('a', 'b'), ('b', 'c')]) == [['a', 'b', 'c']]


def test_native_pixel_cap_before_calibration(module, monkeypatch):
    monkeypatch.setattr(module, 'build_image_derived_session', lambda *_: pytest.fail('calibration before pixel cap'))
    with pytest.raises(ValueError, match='pixel budget'):
        module.run_pixel_ab({'a': np.zeros((2001, 3000, 3), np.uint8), 'b': np.zeros((2, 2, 3), np.uint8)})


def _truth_fixture(tmp_path):
    from mc_photogrammetry.reconstruction.camera import CameraModel
    cameras = [CameraModel((0., 0., 0.), 0., 0., eye_height=0.),
               CameraModel((1., 0., 0.), 0., 0., eye_height=0.)]
    capture = {'cameras': [camera.to_dict() for camera in cameras],
               'frames': [{'filename': f'{index}.png', 'actual': {
                   'eye_position': list(camera.position), 'yaw_degrees': 0., 'pitch_degrees': 0.,
                   'fov_degrees': 60., 'roll_degrees': 0., 'screenshot_sha256': str(index) * 64, 'screenshot_bytes': 99,
                   'framebuffer_width': 96, 'framebuffer_height': 64}}
                          for index, camera in enumerate(cameras)],
               'screenshots': [{'filename': f'{index}.png', 'sha256': str(index) * 64} for index in range(2)]}
    capture_path, ground_path = tmp_path / 'capture.json', tmp_path / 'ground.json'
    capture_path.write_text(json.dumps(capture))
    ground_path.write_text('{}')
    return capture_path, ground_path, cameras


@pytest.mark.parametrize('wrong_rotation,expected_correct', [(False, True), (True, False)])
def test_truth_pose_false_accept_and_source_join(module, monkeypatch, tmp_path, wrong_rotation, expected_correct):
    from mc_photogrammetry.pipeline.contracts import CameraEstimate, CameraPose
    from mc_photogrammetry.reconstruction.camera import camera_rotation_matrix, intrinsics_for, CameraModel
    from mc_photogrammetry.dataset import blocks
    capture, ground, cameras = _truth_fixture(tmp_path)
    monkeypatch.setattr(blocks, 'parse_ground_truth', lambda value: value)
    monkeypatch.setattr(blocks, 'volume_from_ground_truth', lambda value: {})
    estimates = {f'image-{index:04d}': CameraEstimate(intrinsics=intrinsics_for(camera, 96, 64),
        pose=CameraPose(position=None, yaw_degrees=0., pitch_degrees=0., roll_degrees=0.))
        for index, camera in enumerate(cameras)}
    rotation = np.eye(3) if not wrong_rotation else camera_rotation_matrix(CameraModel((0., 0., 0.), 90., 0., eye_height=0.)) @ camera_rotation_matrix(cameras[0]).T
    result = SimpleNamespace(matches=np.empty((0, 4)), solved=True,
        pose=SimpleNamespace(relative_rotation=rotation, baseline_direction=(1., 0., 0.)))
    pair = ('image-0000', 'image-0001')
    report = {'frames': [{'frame_id': name, 'shape': [64, 96, 3]} for name in pair],
              'input_seals': [{'path': f'{index}.png', 'bytes': 99, 'sha256': str(index) * 64} for index in range(2)],
              'pairs': [{'arm': arm, 'left_id': pair[0], 'right_id': pair[1], 'status': 'solved', 'reason': ''}
                        for arm in ('baseline', 'spatial')]}
    module.evaluate_truth(report, {arm: {pair: result} for arm in ('baseline', 'spatial')},
                          SimpleNamespace(cameras=estimates), capture, ground)
    assert report['pairs'][0]['correct_accepted'] is expected_correct
    assert report['summary']['spatial']['false_accepted_pairs'] == (0 if expected_correct else 1)
    assert report['truth_policy']['visible_block_geometry_F1'] is None
    assert report['truth_policy']['exact_state_match'] is None
    assert report['pairs'][0]['truth_geometry']['same_surface_precision'] is None
    report['input_seals'][0]['sha256'] = 'a' * 64
    with pytest.raises(ValueError, match='source join'):
        module.evaluate_truth(report, {arm: {pair: result} for arm in ('baseline', 'spatial')},
                              SimpleNamespace(cameras=estimates), capture, ground)


def test_original_baseline_detector_census_parity_and_single_call(module, monkeypatch):
    import cv2
    from mc_photogrammetry.pipeline.correspondence import CorrespondenceSettings, detect_features
    image = np.random.default_rng(987).integers(0, 256, (180, 240, 3), dtype=np.uint8)
    mask = np.ones((180, 240), bool)
    mask[60:120, 80:160] = False
    settings = CorrespondenceSettings(max_features=96, mask_boundary_pad_px=8.)
    expected = detect_features('a', image, usable_mask=mask, settings=settings)
    original, calls = cv2.SIFT_create, []
    class Recording:
        def __init__(self, detector):
            self.detector = detector
        def detectAndCompute(self, *args):
            calls.append('detect')
            return self.detector.detectAndCompute(*args)
    def factory(*args, **kwargs):
        return Recording(original(*args, **kwargs))
    monkeypatch.setattr(cv2, 'SIFT_create', factory)
    actual, census = module._detect_baseline_with_census('a', image, usable_mask=mask, settings=settings)
    assert cv2.SIFT_create is factory
    assert calls == ['detect']
    np.testing.assert_array_equal(actual.points, expected.points)
    np.testing.assert_array_equal(actual.descriptors, expected.descriptors)
    module._validate_census(census, actual, 'baseline')
    assert census['raw_sift_proposals']['count'] == len(actual.points) + actual.boundary_dropped
    assert census['footprint_retained']['count'] == census['quota_selected']['count'] == census['descriptor_returned']['count']


def test_baseline_census_restores_factory_on_error(module, monkeypatch):
    import cv2
    original = cv2.SIFT_create
    monkeypatch.setattr(module, 'detect_features', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('probe error')))
    with pytest.raises(RuntimeError, match='probe error'):
        module._detect_baseline_with_census('a', np.zeros((10, 10, 3), np.uint8),
            usable_mask=np.ones((10, 10), bool), settings=None)
    assert cv2.SIFT_create is original


@pytest.mark.parametrize('defect', ['bool', 'sum', 'growth', 'source', 'drops'])
def test_census_refuses_nonconserving_or_foreign_stages(module, defect):
    from mc_photogrammetry.pipeline.correspondence import FrameFeatures
    feature = FrameFeatures('a', np.asarray([[10., 10.]]), np.ones((1, 128), np.float32), 96, 64, 96 * 64)
    census = {stage: module._stage_census(feature.points, 96, 64) for stage in
        ('raw_sift_proposals', 'footprint_retained', 'quota_selected', 'descriptor_returned')}
    if defect == 'bool':
        census['raw_sift_proposals']['count'] = True
    elif defect == 'sum':
        census['quota_selected']['cells'][0] = 2
    elif defect == 'growth':
        census['quota_selected'] = {'count': 2, 'cells': [2] + [0] * 15}
    elif defect == 'source':
        census['descriptor_returned'] = {'count': 0, 'cells': [0] * 16}
    else:
        feature = FrameFeatures('a', feature.points, feature.descriptors, 96, 64, 96 * 64, boundary_dropped=1)
    with pytest.raises(ValueError, match='census'):
        module._validate_census(census, feature, 'spatial')


def _empty_report():
    return {'frames': [{'frame_id': f'image-{index:04d}', 'shape': [64, 96, 3]} for index in range(2)], 'pairs': []}


@pytest.mark.parametrize('defect', ['missing', 'eye', 'eye_bool', 'yaw_bool', 'yaw_nan', 'pitch', 'fov', 'fov_missing', 'dimensions', 'roll', 'zero_baseline', 'duplicate'])
def test_invalid_actual_truth_never_falls_back_to_valid_planned(module, monkeypatch, tmp_path, defect):
    capture, ground, _ = _truth_fixture(tmp_path)
    data = json.loads(capture.read_bytes())
    actual = data['frames'][0]['actual']
    if defect == 'missing':
        del data['frames'][0]['actual']
    elif defect == 'eye':
        actual['eye_position'] = [0., 0.]
    elif defect == 'eye_bool':
        actual['eye_position'][0] = True
    elif defect == 'yaw_bool':
        actual['yaw_degrees'] = True
    elif defect == 'yaw_nan':
        actual['yaw_degrees'] = float('nan')
    elif defect == 'pitch':
        actual['pitch_degrees'] = 91.
    elif defect == 'fov':
        actual['fov_degrees'] = 180.
    elif defect == 'fov_missing':
        del actual['fov_degrees']
    elif defect == 'dimensions':
        actual['framebuffer_width'] = True
    elif defect == 'roll':
        del actual['roll_degrees']
    elif defect == 'zero_baseline':
        data['frames'][1]['actual']['eye_position'] = actual['eye_position']
    else:
        data['frames'][1]['filename'] = data['frames'][0]['filename']
    capture.write_text(json.dumps(data))
    # Ground truth parsing must not precede actual-pose validation.
    from mc_photogrammetry.dataset import blocks
    monkeypatch.setattr(blocks, 'parse_ground_truth', lambda *_: pytest.fail('invalid actual pose admitted'))
    with pytest.raises(ValueError):
        module.evaluate_truth(_empty_report(), {'baseline': {}, 'spatial': {}}, SimpleNamespace(cameras={}), capture, ground)


def test_exact_consumed_truth_bytes_hashed_once_despite_later_mutation(module, monkeypatch, tmp_path):
    from mc_photogrammetry.dataset import blocks
    capture, ground, _ = _truth_fixture(tmp_path)
    expected_capture, expected_ground = capture.read_bytes(), ground.read_bytes()
    original_read, reads = Path.read_bytes, []
    def read_and_mutate(path):
        raw = original_read(path)
        reads.append(path)
        path.write_bytes(b'CHANGED AFTER SNAPSHOT')
        return raw
    monkeypatch.setattr(Path, 'read_bytes', read_and_mutate)
    monkeypatch.setattr(blocks, 'parse_ground_truth', lambda value: value)
    monkeypatch.setattr(blocks, 'volume_from_ground_truth', lambda value: {})
    report = _empty_report()
    module.evaluate_truth(report, {'baseline': {}, 'spatial': {}}, SimpleNamespace(cameras={}), capture, ground)
    assert reads == [capture, ground]
    assert report['evaluation_capture_sha256'] == hashlib.sha256(expected_capture).hexdigest()
    assert report['evaluation_ground_truth_sha256'] == hashlib.sha256(expected_ground).hexdigest()
    assert report['truth_policy']['planned_camera_fallback'] is False


def test_nonzero_camera_frame_direction_uses_actual_not_planned(module, monkeypatch, tmp_path):
    from mc_photogrammetry.dataset import blocks
    from mc_photogrammetry.pipeline.contracts import CameraEstimate, CameraPose
    from mc_photogrammetry.reconstruction.camera import CameraModel, camera_rotation_matrix, intrinsics_for
    capture, ground, _ = _truth_fixture(tmp_path)
    data = json.loads(capture.read_bytes())
    cameras = [CameraModel((0., 0., 0.), 17., 9., roll_degrees=3., eye_height=0.),
               CameraModel((1., .2, .1), 28., -4., roll_degrees=-2., eye_height=0.)]
    for frame, camera in zip(data['frames'], cameras):
        frame['actual'].update(eye_position=list(camera.position), yaw_degrees=camera.yaw_degrees,
                              pitch_degrees=camera.pitch_degrees, roll_degrees=camera.roll_degrees)
    data['cameras'] = [None, None]  # A valid actual record does not need a planned record.
    capture.write_text(json.dumps(data))
    monkeypatch.setattr(blocks, 'parse_ground_truth', lambda value: value)
    monkeypatch.setattr(blocks, 'volume_from_ground_truth', lambda value: {})
    estimates = {f'image-{index:04d}': CameraEstimate(intrinsics=intrinsics_for(camera, 96, 64),
        pose=CameraPose(position=None, yaw_degrees=camera.yaw_degrees, pitch_degrees=camera.pitch_degrees,
                       roll_degrees=camera.roll_degrees)) for index, camera in enumerate(cameras)}
    direction = np.asarray(cameras[1].position) - np.asarray(cameras[0].position)
    direction /= np.linalg.norm(direction)
    result = SimpleNamespace(matches=np.empty((0, 4)), solved=True,
        pose=SimpleNamespace(relative_rotation=camera_rotation_matrix(cameras[1]) @ camera_rotation_matrix(cameras[0]).T,
                             baseline_direction=tuple(direction)))
    pair = ('image-0000', 'image-0001')
    report = _empty_report()
    report['pairs'] = [{'arm': arm, 'left_id': pair[0], 'right_id': pair[1], 'status': 'solved', 'reason': ''} for arm in ('baseline', 'spatial')]
    module.evaluate_truth(report, {arm: {pair: result} for arm in ('baseline', 'spatial')}, SimpleNamespace(cameras=estimates), capture, ground)
    assert report['summary']['spatial']['correct_accepted_pairs'] == 1
    assert report['pairs'][0]['pose_error']['baseline_degrees'] < 1e-5
    assert report['pairs'][0]['pose_error']['rotation_degrees'] < 1e-5


@pytest.mark.parametrize('support_identity', ['historical_feature_rows', 'physical_pixels'])
def test_actual_frozen_planar_wrong_pose_is_scored_false_accepted(module, monkeypatch, tmp_path, support_identity):
    # Load the existing evaluator-generated raster, not any screenshot payload.
    path = Path(__file__).with_name('test_spatial_matching.py')
    spec = importlib.util.spec_from_file_location('spatial_frozen_falsifier', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    from mc_photogrammetry.pipeline.spatial_matching import match_and_solve_spatial_pair
    from mc_photogrammetry.dataset import blocks
    from mc_photogrammetry.domain.models import BlockState
    images, estimates = helper._projected_texture_pair()
    pair = ('image-0000', 'image-0001')
    # Retain the archived counterexample under both original row identity AND
    # raw-edge shortlist semantics. This controlled replay is not an execution
    # of the entire archived source. Current physical support now refuses.
    with monkeypatch.context() as historical:
        if support_identity == 'historical_feature_rows':
            import mc_photogrammetry.pipeline.joint_matching as matcher_module
            historical.setattr('mc_photogrammetry.pipeline.joint_matching._pixel_endpoint_ids',
                               lambda points: np.arange(len(points)))
            original_greedy = matcher_module._greedy_selection
            original_refit = matcher_module._select_and_refit

            def raw_row_shortlist(rows, norms, graph, t, threshold, *, endpoint_ids=None):
                return matcher_module._residuals(rows, norms, t) <= threshold

            def row_identity_refit(*args, **kwargs):
                with monkeypatch.context() as refit:
                    refit.setattr(matcher_module, '_greedy_selection', original_greedy)
                    return original_refit(*args, **kwargs)

            historical.setattr(matcher_module, '_greedy_selection', raw_row_shortlist)
            historical.setattr(matcher_module, '_select_and_refit', row_identity_refit)
        candidate = match_and_solve_spatial_pair(pair[0], images[0], pair[1], images[1],
            left_camera=estimates[0], right_camera=estimates[1],
            usable_masks={name: np.ones((480, 640), bool) for name in pair})
    if support_identity == 'historical_feature_rows':
        assert candidate.solved and candidate.pose.right_yaw_class == 1
    else:
        assert not candidate.solved and candidate.reason == 'ambiguous_hypothesis'
        assert candidate.pose is None
    capture, ground, _ = _truth_fixture(tmp_path)
    data = json.loads(capture.read_bytes())
    for frame, eye, yaw, pitch in zip(data['frames'], ((0., 0., 0.), (1.5, .2, .1)), (4., 10.), (2., -1.)):
        frame['actual'].update(eye_position=list(eye), yaw_degrees=yaw, pitch_degrees=pitch,
            fov_degrees=75., framebuffer_width=640, framebuffer_height=480)
    capture.write_text(json.dumps(data))
    state = BlockState('minecraft:stone')
    monkeypatch.setattr(blocks, 'parse_ground_truth', lambda value: value)
    monkeypatch.setattr(blocks, 'volume_from_ground_truth', lambda value: {
        (x, y, 12): SimpleNamespace(state=state) for x in range(-10, 11) for y in range(-7, 8)})
    report = {'frames': [{'frame_id': name, 'shape': [480, 640, 3]} for name in pair],
              'pairs': [{'arm': 'spatial', 'left_id': pair[0], 'right_id': pair[1],
                         'status': candidate.status, 'reason': candidate.reason}]}
    module.evaluate_truth(report, {'baseline': {}, 'spatial': {pair: candidate}},
        SimpleNamespace(cameras=dict(zip(pair, estimates))), capture, ground)
    scored = report['pairs'][0]
    if support_identity == 'historical_feature_rows':
        assert scored['pose_error']['baseline_degrees'] == pytest.approx(48.967702402206115, abs=1e-8)
        assert scored['pose_error']['rotation_degrees'] == pytest.approx(90., abs=1e-8)
        assert report['summary']['spatial']['accepted_pairs'] == 1
        assert report['summary']['spatial']['false_accepted_pairs'] == 1
    else:
        assert scored['pose_error'] is None
        assert report['summary']['spatial']['accepted_pairs'] == 0
        assert report['summary']['spatial']['false_accepted_pairs'] == 0
        assert report['summary']['spatial']['refusals'] == {'ambiguous_hypothesis': 1}
        assert len(np.unique(candidate.matches[:, :2], axis=0)) == candidate.support
        assert len(np.unique(candidate.matches[:, 2:], axis=0)) == candidate.support
    assert scored['correct_accepted'] is False
    assert report['summary']['spatial']['correct_accepted_pairs'] == 0
    assert scored['truth_geometry']['unique_selected'] > 0


def _camera_family_fixture(module, monkeypatch, tmp_path, *, bias=0., right_gauge=0., world_gauge=0.):
    """Actual sparse geometry, independently specified estimate bias/gauges."""
    from mc_photogrammetry.dataset import blocks
    from mc_photogrammetry.pipeline.contracts import CameraEstimate, CameraPose
    from mc_photogrammetry.pipeline.joint_matching import match_feature_frames
    from mc_photogrammetry.reconstruction.camera import CameraModel
    spec = importlib.util.spec_from_file_location('camera_family_sparse_scene', Path(__file__).with_name('test_joint_matching.py'))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    left, right, left_est, right_est, left_truth, right_truth = helper._scene()
    estimates, actuals = {}, []
    for index, (estimate, actual) in enumerate(zip((left_est, right_est), (left_truth, right_truth))):
        name = f'image-{index:04d}'
        estimates[name] = CameraEstimate(intrinsics=estimate.intrinsics, pose=CameraPose(
            position=None, yaw_degrees=estimate.pose.yaw_degrees + world_gauge + (right_gauge if index else 0.),
            pitch_degrees=estimate.pose.pitch_degrees + (bias if index else 0.), roll_degrees=estimate.pose.roll_degrees))
        position = module._quarter_turn(world_gauge / 90.).T @ np.asarray(actual.position)
        actuals.append(CameraModel(tuple(position), actual.yaw_degrees + world_gauge,
                                  actual.pitch_degrees, eye_height=0., roll_degrees=actual.roll_degrees))
    # Candidate selection sees the same observed points/descriptors and estimates,
    # never actuals or their comparison metrics. Truth enters only below.
    result = match_feature_frames(left, right, estimates['image-0000'], estimates['image-0001'])
    solution = SimpleNamespace(cameras=estimates, session_sha256='a' * 64)
    report = {'session_sha256': solution.session_sha256,
        'frames': [{'frame_id': name, 'shape': [720, 1280, 3],
                    'image_camera': module._camera_snapshot(name, estimate, solution.session_sha256)}
                   for name, estimate in estimates.items()],
        'pairs': [{'arm': arm, 'left_id': 'image-0000', 'right_id': 'image-0001',
                   'status': result.status, 'reason': result.reason} for arm in ('baseline', 'spatial')]}
    capture = tmp_path / 'camera-family-capture.json'
    ground = tmp_path / 'camera-family-ground.json'
    capture.write_text(json.dumps({'frames': [{'filename': f'{index}.png', 'actual': {
        'eye_position': list(camera.position), 'yaw_degrees': camera.yaw_degrees,
        'pitch_degrees': camera.pitch_degrees, 'roll_degrees': camera.roll_degrees,
        'fov_degrees': camera.fov_degrees, 'framebuffer_width': 1280, 'framebuffer_height': 720}}
        for index, camera in enumerate(actuals)]}))
    ground.write_text('{}')
    monkeypatch.setattr(blocks, 'parse_ground_truth', lambda value: value)
    monkeypatch.setattr(blocks, 'volume_from_ground_truth', lambda value: {})
    results = {arm: {('image-0000', 'image-0001'): result} for arm in ('baseline', 'spatial')}
    before = result.matches.copy()
    module.evaluate_truth(report, results, solution, capture, ground)
    np.testing.assert_array_equal(result.matches, before)
    return report, result, solution, capture, ground


@pytest.mark.parametrize('bias', [0., 12.])
def test_camera_family_exact_sparse_and_12_degree_bias(module, monkeypatch, tmp_path, bias):
    report, result, *_ = _camera_family_fixture(module, monkeypatch, tmp_path, bias=bias)
    assert report['frames'][0]['calibration_truth_error']['minimum_gauged_rotation_error_degrees'] < 2e-6
    right = report['frames'][1]['calibration_truth_error']
    assert right['pitch_error_degrees'] == bias
    assert right['yaw_error_mod90_degrees'] == 0.
    assert right['minimum_gauged_rotation_error_degrees'] == pytest.approx(bias, abs=2e-6)
    assert right['focal_error_relative'] == {'fx': 0., 'fy': 0.}
    for row in report['pairs']:
        family = row['camera_family_truth_error']
        assert len(family['rotation_errors_by_yaw_class_degrees']) == 4
        assert family['best_yaw_class'] == 1
        assert family['minimum_rotation_error_degrees'] == pytest.approx(bias, abs=2e-6)
        assert family['contains_rotation_within_5_degrees'] is (bias == 0.)
    if bias == 0.:
        assert result.solved and report['summary']['baseline']['correct_accepted_pairs'] == 1
    else:
        assert min(report['pairs'][0]['camera_family_truth_error']['rotation_errors_by_yaw_class_degrees']) > 5.
        assert report['summary']['baseline']['correct_accepted_pairs'] == 0


def test_camera_family_90_degree_relabel_and_common_world_y_gauge(module, monkeypatch, tmp_path):
    ordinary, *_ = _camera_family_fixture(module, monkeypatch, tmp_path)
    relabelled, *_ = _camera_family_fixture(module, monkeypatch, tmp_path, right_gauge=90.)
    common, *_ = _camera_family_fixture(module, monkeypatch, tmp_path, world_gauge=37.)
    base_errors = ordinary['pairs'][0]['camera_family_truth_error']['rotation_errors_by_yaw_class_degrees']
    relabel_errors = relabelled['pairs'][0]['camera_family_truth_error']['rotation_errors_by_yaw_class_degrees']
    np.testing.assert_allclose(relabel_errors, np.roll(base_errors, -1), atol=2e-6)
    assert relabelled['pairs'][0]['camera_family_truth_error']['best_yaw_class'] == 0
    assert relabelled['frames'][1]['calibration_truth_error']['yaw_error_mod90_degrees'] == 0.
    np.testing.assert_allclose(common['pairs'][0]['camera_family_truth_error']['rotation_errors_by_yaw_class_degrees'], base_errors, atol=2e-6)
    assert common['summary']['baseline']['correct_accepted_pairs'] == 1


def test_camera_telemetry_signed_units_roll_assumption_and_refused_pairs(module, monkeypatch, tmp_path):
    from mc_photogrammetry.dataset import blocks
    from mc_photogrammetry.pipeline.contracts import CameraEstimate, CameraPose, CameraIntrinsics
    from mc_photogrammetry.reconstruction.camera import intrinsics_for, CameraModel
    capture, ground, cameras = _truth_fixture(tmp_path)
    data = json.loads(capture.read_bytes())
    for frame in data['frames']:
        frame['actual'].pop('roll_degrees')
        frame['actual'].update(shader_pack_applied=False, active_resource_packs=['vanilla'])
    capture.write_text(json.dumps(data))
    optics = intrinsics_for(CameraModel(cameras[0].position, 0., 0., fov_degrees=60.), 96, 64)
    estimates = {f'image-{i:04d}': CameraEstimate(pose=CameraPose(None, 45., -12., 3.),
        intrinsics=CameraIntrinsics(optics.image_size,
            (optics.focal_length_pixels[0] * 1.1, optics.focal_length_pixels[1] * .9),
            optics.principal_point_pixels)) for i in range(2)}
    report = _empty_report()
    report['pairs'] = [{'arm': arm, 'left_id': 'image-0000', 'right_id': 'image-0001',
                        'status': 'refused', 'reason': 'insufficient_one_to_one_support'}
                       for arm in ('baseline', 'spatial')]
    result = SimpleNamespace(matches=np.empty((0, 4)), solved=False)
    monkeypatch.setattr(blocks, 'parse_ground_truth', lambda value: value)
    monkeypatch.setattr(blocks, 'volume_from_ground_truth', lambda value: {})
    module.evaluate_truth(report, {arm: {('image-0000', 'image-0001'): result} for arm in ('baseline', 'spatial')},
                          SimpleNamespace(cameras=estimates), capture, ground)
    error = report['frames'][0]['calibration_truth_error']
    assert error['yaw_error_mod90_degrees'] == -45.  # Declared nearest-residual tie.
    assert error['pitch_error_degrees'] == -12.
    assert error['roll_error_degrees'] == 3.
    assert error['roll_truth_source'] == 'vanilla_shader_free_engine_zero_roll_assumption'
    assert error['focal_error_relative']['fx'] == pytest.approx(.1)
    assert error['focal_error_relative']['fy'] == pytest.approx(-.1)
    assert all(row['camera_family_truth_error']['status'] == 'available' for row in report['pairs'])
    assert all(row['pose_error'] is None for row in report['pairs'])


@pytest.mark.parametrize('defect', ['frame', 'session', 'matrix', 'report_session'])
def test_camera_snapshot_source_binding_refuses_tampering(module, monkeypatch, tmp_path, defect):
    report, result, solution, capture, ground = _camera_family_fixture(module, monkeypatch, tmp_path)
    snapshot = report['frames'][0]['image_camera']
    if defect == 'frame':
        snapshot['frame_id'] = 'foreign'
    elif defect == 'session':
        snapshot['session_sha256'] = 'b' * 64
    elif defect == 'matrix':
        snapshot['world_to_camera_rows'][0][0] += .01
    else:
        report['session_sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='join mismatch'):
        module.evaluate_truth(report, {arm: {('image-0000', 'image-0001'): result} for arm in ('baseline', 'spatial')},
                              solution, capture, ground)


def test_camera_truth_telemetry_only_after_all_30_matching_calls(module, monkeypatch, tmp_path):
    from mc_photogrammetry.dataset import blocks
    from mc_photogrammetry.pipeline.contracts import CameraEstimate, CameraPose
    from mc_photogrammetry.pipeline.correspondence import FrameFeatures
    from mc_photogrammetry.reconstruction.camera import CameraModel, intrinsics_for
    frames = {f'image-{index:04d}': np.zeros((64, 96, 3), np.uint8) for index in range(6)}
    truth = [CameraModel((float(index), 0., 0.), 0., 0., fov_degrees=60., eye_height=0.) for index in range(6)]
    solution = SimpleNamespace(session_sha256='c' * 64, cameras={name: CameraEstimate(
        intrinsics=intrinsics_for(camera, 96, 64), pose=CameraPose(None, 0., 0., 0.))
        for name, camera in zip(frames, truth)})
    monkeypatch.setattr(module, 'build_image_derived_session', lambda frames: solution)

    def detector(name, image, **kwargs):
        feature = FrameFeatures(name, np.empty((0, 2)), np.empty((0, 128)), 96, 64,
                                int(kwargs['usable_mask'].sum()))
        census = {stage: module._stage_census(feature.points, 96, 64) for stage in
            ('raw_sift_proposals', 'footprint_retained', 'quota_selected', 'descriptor_returned')}
        if '_census' in kwargs:
            kwargs['_census'].update(census)
            return feature
        return feature, census

    monkeypatch.setattr(module, '_detect_baseline_with_census', detector)
    monkeypatch.setattr(module, 'detect_spatial_features', detector)
    original_match, matches, reads = module.match_feature_frames, [], []

    def match(*args, **kwargs):
        assert not reads
        result = original_match(*args, **kwargs)
        matches.append(result)
        return result

    monkeypatch.setattr(module, 'match_feature_frames', match)
    original_snapshot = module._truth_snapshot

    def snapshot(path, cap):
        assert len(matches) == 30
        reads.append(path)
        return original_snapshot(path, cap)

    monkeypatch.setattr(module, '_truth_snapshot', snapshot)
    report, results, session = module.run_pixel_ab(frames)
    assert len(matches) == 30 and not reads
    assert all('calibration_truth_error' not in row for row in report['frames'])
    camera_records = json.dumps([row['image_camera'] for row in report['frames']], sort_keys=True)
    statuses = [(row['status'], row['reason']) for row in report['pairs']]
    capture, ground = tmp_path / 'six-capture.json', tmp_path / 'six-ground.json'
    capture.write_text(json.dumps({'frames': [{'filename': f'{i}.png', 'actual': {
        'eye_position': list(camera.position), 'yaw_degrees': 0., 'pitch_degrees': 0.,
        'roll_degrees': 0., 'fov_degrees': 60., 'framebuffer_width': 96, 'framebuffer_height': 64}}
        for i, camera in enumerate(truth)]}))
    ground.write_text('{}')
    monkeypatch.setattr(blocks, 'parse_ground_truth', lambda value: value)
    monkeypatch.setattr(blocks, 'volume_from_ground_truth', lambda value: {})
    module.evaluate_truth(report, results, session, capture, ground)
    assert reads == [capture, ground]
    assert [(row['status'], row['reason']) for row in report['pairs']] == statuses
    assert json.dumps([row['image_camera'] for row in report['frames']], sort_keys=True) == camera_records
    assert all(row['camera_family_truth_error']['minimum_rotation_error_degrees'] < 2e-6 for row in report['pairs'])
    assert all(row['camera_family_truth_error']['contains_rotation_within_5_degrees'] for row in report['pairs'])


def test_missing_image_camera_telemetry_is_unavailable_not_zero(module, monkeypatch, tmp_path):
    from mc_photogrammetry.dataset import blocks
    capture, ground, _ = _truth_fixture(tmp_path)
    monkeypatch.setattr(blocks, 'parse_ground_truth', lambda value: value)
    monkeypatch.setattr(blocks, 'volume_from_ground_truth', lambda value: {})
    report = _empty_report()
    report['pairs'] = [{'arm': 'baseline', 'left_id': 'image-0000', 'right_id': 'image-0001',
                        'status': 'refused', 'reason': 'invalid_input'}]
    module.evaluate_truth(report, {'baseline': {}, 'spatial': {}}, SimpleNamespace(cameras={}), capture, ground)
    assert report['frames'][0]['calibration_truth_error'] == {
        'status': 'unavailable', 'reason': 'image_camera_unavailable'}
    assert report['pairs'][0]['camera_family_truth_error'] == {
        'status': 'unavailable', 'reason': 'image_camera_unavailable'}
