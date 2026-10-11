"""Bounded baseline versus spatial detector A/B; truth is read after matching.

Neither production matcher receives filenames, manifests, truth poses or blocks.
Default production is unchanged. This reports pair geometry, not delivered
visible-block F1/state accuracy. One development scene is not a holdout.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys
import time

# Direct CLI execution must import this checkout, independently of cwd, pytest
# pythonpath, editable-install .pth files, or caller environment variables.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

import numpy as np

from mc_photogrammetry.pipeline.correspondence import CorrespondenceSettings, detect_features, usable_mask_for
from mc_photogrammetry.pipeline.image_derived_session import build_image_derived_session
from mc_photogrammetry.pipeline.joint_matching import JointMatchSettings, match_feature_frames
from mc_photogrammetry.pipeline.connected_camera import PairGates
from mc_photogrammetry.pipeline.spatial_matching import detect_spatial_features, _stage_census
from mc_photogrammetry.reconstruction.camera import CameraModel, camera_rotation_matrix, intrinsics_for


def cells(points, width, height):
    bins = np.clip((points[:, 0] * 4 / width).astype(int), 0, 3) + 4 * np.clip((points[:, 1] * 4 / height).astype(int), 0, 3)
    return np.bincount(bins, minlength=16).tolist()


def _detect_baseline_with_census(frame_id, image, *, usable_mask, settings):
    """Observe the original SIFT call once; restore the synchronous probe always.

    Baseline raw domain is AFTER OpenCV's global strongest-feature cap and mask.
    No new quota is applied after footprint filtering. Descriptor-returned is
    the retained production detector API result (native compute precedes filter).
    """
    import cv2
    original = cv2.SIFT_create
    raw = None
    class Probe:
        def __init__(self, detector):
            self.detector = detector
        def detectAndCompute(self, *args, **kwargs):
            nonlocal raw
            keypoints, descriptors = self.detector.detectAndCompute(*args, **kwargs)
            if raw is not None:
                raise ValueError('duplicate baseline detector call')
            points = np.asarray([point.pt for point in keypoints or ()], dtype=float).reshape(-1, 2)
            if len(points) > 64000:
                raise ValueError('baseline detector census budget')
            raw = _stage_census(points, image.shape[1], image.shape[0])
            return keypoints, descriptors
    cv2.SIFT_create = lambda *args, **kwargs: Probe(original(*args, **kwargs))
    try:
        feature = detect_features(frame_id, image, usable_mask=usable_mask, settings=settings)
    finally:
        cv2.SIFT_create = original
    if raw is None:
        raise ValueError('baseline raw detector census absent')
    final = _stage_census(feature.points, feature.width, feature.height)
    return feature, {'raw_sift_proposals': raw, 'footprint_retained': final,
                     'quota_selected': dict(final), 'descriptor_returned': dict(final)}


def _validate_census(census, feature, arm):
    stages = ('raw_sift_proposals', 'footprint_retained', 'quota_selected', 'descriptor_returned')
    prior = None
    for stage in stages:
        row = census.get(stage)
        if type(row) is not dict or type(row.get('count')) is not int or not 0 <= row['count'] <= 64000:
            raise ValueError('feature census count')
        bins = row.get('cells')
        if type(bins) is not list or len(bins) != 16 or any(type(value) is not int or value < 0 for value in bins) or sum(bins) != row['count']:
            raise ValueError('feature census cell conservation')
        if prior is not None and any(value > old for value, old in zip(bins, prior)):
            raise ValueError('feature census stage conservation')
        prior = bins
    if census['descriptor_returned'] != _stage_census(feature.points, feature.width, feature.height):
        raise ValueError('descriptor census source mismatch')
    if len(feature.points) != len(feature.descriptors):
        raise ValueError('descriptor census row mismatch')
    if census['raw_sift_proposals']['count'] - census['footprint_retained']['count'] != feature.boundary_dropped:
        raise ValueError('footprint census drop conservation')
    if arm == 'spatial' and (census['quota_selected']['count'] > 4000 or max(census['quota_selected']['cells']) > 250):
        raise ValueError('feature quota census cap')


def _rotation(camera):
    pose = camera.pose
    return camera_rotation_matrix(CameraModel((0., 0., 0.), pose.yaw_degrees,
        pose.pitch_degrees, roll_degrees=pose.roll_degrees, eye_height=0.))


def _quarter_turn(yaw_class):
    angle = math.radians(90. * yaw_class)
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((c, 0., s), (0., 1., 0.), (-s, 0., c)))


def _rotation_error(actual, estimated):
    return float(np.degrees(np.arccos(np.clip(
        (np.trace(actual.T @ estimated) - 1.) / 2., -1., 1.))))


def _camera_snapshot(frame_id, estimate, session_sha256):
    """Same-pass estimate only; never consult evaluation truth or refit cameras."""
    result = {'frame_id': frame_id, 'session_sha256': session_sha256,
              'status': 'unavailable', 'reason': 'image_camera_unavailable'}
    if estimate is None or estimate.pose is None or estimate.intrinsics is None:
        return result
    pose, optics = estimate.pose, estimate.intrinsics
    result.update(status='available', reason='',
        yaw_degrees=float(pose.yaw_degrees), pitch_degrees=float(pose.pitch_degrees),
        roll_degrees=float(pose.roll_degrees),
        image_size=list(optics.image_size), focal_length_pixels=list(optics.focal_length_pixels),
        principal_point_pixels=list(optics.principal_point_pixels),
        world_to_camera_rows=_rotation(estimate).tolist(),
        estimator_confidence=float(estimate.confidence),
        estimator_residual_pixels=None if estimate.residual_pixels is None else float(estimate.residual_pixels),
        estimate_scope='single_view_orientation_mod90_and_intrinsics_no_translation_or_scale_claim')
    numbers = [result[key] for key in ('yaw_degrees', 'pitch_degrees', 'roll_degrees')]
    numbers += result['focal_length_pixels'] + result['principal_point_pixels']
    numbers.append(result['estimator_confidence'])
    if result['estimator_residual_pixels'] is not None:
        numbers.append(result['estimator_residual_pixels'])
    if not all(math.isfinite(value) for value in numbers) or min(result['focal_length_pixels']) <= 0:
        raise ValueError('image camera snapshot invalid')
    return result


def _frame_calibration_error(snapshot, actual, roll_source, shape):
    if snapshot['status'] != 'available':
        return {'status': 'unavailable', 'reason': snapshot['reason']}
    if snapshot['image_size'] != [shape[1], shape[0]]:
        raise ValueError('image camera snapshot dimensions mismatch')
    estimated = np.asarray(snapshot['world_to_camera_rows'])
    truth_rotation = camera_rotation_matrix(actual)
    errors = [_rotation_error(truth_rotation, estimated @ _quarter_turn(k)) for k in range(4)]
    best = int(np.argmin(errors))
    truth_optics = intrinsics_for(actual, shape[1], shape[0])
    return {'status': 'available',
        'yaw_error_mod90_degrees': (snapshot['yaw_degrees'] - actual.yaw_degrees + 45.) % 90. - 45.,
        'pitch_error_degrees': snapshot['pitch_degrees'] - actual.pitch_degrees,
        'roll_error_degrees': (snapshot['roll_degrees'] - actual.roll_degrees + 180.) % 360. - 180.,
        'roll_truth_source': roll_source,
        'focal_error_relative': {axis: (estimated_focal - true_focal) / true_focal
            for axis, estimated_focal, true_focal in zip(('fx', 'fy'),
                snapshot['focal_length_pixels'], truth_optics.focal_length_pixels, strict=True)},
        'rotation_errors_by_world_y_quarter_turn_degrees': errors,
        'best_world_y_quarter_turn': best, 'minimum_gauged_rotation_error_degrees': errors[best]}


def _pair_camera_family_error(left, right, actual_left, actual_right):
    if left['status'] != 'available' or right['status'] != 'available':
        return {'status': 'unavailable', 'reason': 'image_camera_unavailable'}
    estimated_left = np.asarray(left['world_to_camera_rows'])
    estimated_right = np.asarray(right['world_to_camera_rows'])
    expected = camera_rotation_matrix(actual_right) @ camera_rotation_matrix(actual_left).T
    errors = [_rotation_error(expected, estimated_right @ _quarter_turn(k) @ estimated_left.T)
              for k in range(4)]
    best = int(np.argmin(errors))
    return {'status': 'available', 'rotation_errors_by_yaw_class_degrees': errors,
        'best_yaw_class': best, 'minimum_rotation_error_degrees': errors[best],
        'contains_rotation_within_5_degrees': errors[best] <= 5.,
        'scope': 'evaluation_only_fixed_image_camera_family_not_selected_pose_or_translation'}


def _angle(first, second):
    first, second = np.asarray(first), np.asarray(second)
    return float(np.degrees(np.arccos(np.clip(np.dot(first, second) /
        (np.linalg.norm(first) * np.linalg.norm(second)), -1., 1.))))


def components(ids, edges):
    groups = {name: {name} for name in ids}
    for left, right in edges:
        merged = groups[left] | groups[right]
        for member in merged:
            groups[member] = merged
    return sorted([sorted(group) for group in {frozenset(group) for group in groups.values()}])


def run_pixel_ab(frames):
    """Pixel-only phase. Cache exact baseline and candidate once per frame."""
    if not 2 <= len(frames) <= 6:
        raise ValueError('A/B requires 2..6 frames')
    for image in frames.values():
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('A/B requires RGB uint8 arrays')
        if image.shape[0] * image.shape[1] > 6_000_000:
            raise ValueError('native pixel budget')
    bounds, gates = JointMatchSettings(), PairGates()
    detection = CorrespondenceSettings(max_features=bounds.max_features,
                                       mask_boundary_pad_px=bounds.mask_boundary_pad_px)
    solution = build_image_derived_session(frames)
    features = {'baseline': {}, 'spatial': {}}
    frame_rows = []
    for frame_id, image in frames.items():
        mask = usable_mask_for(frame_id, image.shape, None)
        row = {'frame_id': frame_id, 'shape': list(image.shape), 'usable_pixels': int(mask.sum()),
               'image_camera': _camera_snapshot(frame_id, solution.cameras.get(frame_id),
                                               solution.session_sha256)}
        for arm in ('baseline', 'spatial'):
            if arm == 'baseline':
                feature, census = _detect_baseline_with_census(frame_id, image, usable_mask=mask, settings=detection)
            else:
                census = {}
                feature = detect_spatial_features(frame_id, image, usable_mask=mask, settings=detection, _census=census)
            _validate_census(census, feature, arm)
            features[arm][frame_id] = feature
            row[arm] = {'features': len(feature.points), 'cells': cells(feature.points, feature.width, feature.height),
                        'boundary_dropped': feature.boundary_dropped, 'stages': census,
                        'raw_domain': 'globally_capped_detector_after_mask' if arm == 'baseline' else 'uncapped_detector_after_mask',
                        'quota_stage': 'pass_through_global_quota_precedes_raw_domain' if arm == 'baseline' else '250_per_cell_after_footprint',
                        'descriptor_stage': 'retained_detector_API_return'}
        frame_rows.append(row)
    results, rows = {}, []
    for arm in features:
        results[arm] = {}
        for left, right in combinations(frames, 2):
            if left not in solution.cameras or right not in solution.cameras:
                rows.append({'arm': arm, 'left_id': left, 'right_id': right, 'status': 'excluded', 'reason': 'camera_unavailable'})
                continue
            start = time.perf_counter()
            result = match_feature_frames(features[arm][left], features[arm][right],
                solution.cameras[left], solution.cameras[right], matcher=bounds, gates=gates)
            results[arm][left, right] = result
            row = {name: getattr(result, name) for name in (
                'status', 'reason', 'raw_keypoints', 'candidate_edges', 'total_graph_edges', 'truncated',
                'ratio_survivors', 'mutual_survivors', 'support', 'support_floor',
                'spatial_bins_left', 'spatial_bins_right', 'median_selected_residual_degrees',
                'parallax_degrees', 'cheirality_fraction', 'singular_ratio', 'ambiguity_margin')}
            row.update(arm=arm, left_id=left, right_id=right, seconds=time.perf_counter() - start,
                selected_cells_left=cells(result.matches[:, :2], features[arm][left].width, features[arm][left].height),
                selected_cells_right=cells(result.matches[:, 2:], features[arm][right].width, features[arm][right].height),
                hypotheses=[asdict(hypothesis) for hypothesis in result.hypotheses])
            rows.append(row)
    return {'frames': frame_rows, 'pairs': rows, 'matcher': asdict(bounds), 'gates': asdict(gates),
            'session_sha256': solution.session_sha256}, results, solution


def _truth_snapshot(path, maximum_bytes):
    """Read once, bound, hash and parse the same immutable bytes."""
    path = Path(path)
    if path.stat().st_size > maximum_bytes:
        raise ValueError('truth snapshot byte budget')
    raw = path.read_bytes()
    if len(raw) > maximum_bytes:
        raise ValueError('truth snapshot byte budget')
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _actual_truth_camera(frame, shape):
    """Actual eye/orientation/FOV only; planned camera records are never read.

    Intrinsics follow the uncropped Minecraft framebuffer's centered,
    square-pixel projection. Missing roll is supported only for an explicitly
    vanilla, shader-free capture, where the engine has zero camera roll.
    """
    actual = frame.get('actual')
    if type(actual) is not dict:
        raise ValueError('actual camera truth missing')
    def number(value):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('actual camera truth must be finite numeric')
        return float(value)
    eye = actual.get('eye_position')
    if type(eye) is not list or len(eye) != 3:
        raise ValueError('actual eye truth missing or invalid')
    position = tuple(number(value) for value in eye)
    yaw = number(actual.get('yaw_degrees'))
    pitch = number(actual.get('pitch_degrees'))
    fov = number(actual.get('fov_degrees'))
    if not -90 <= pitch <= 90 or not 1 <= fov < 180:
        raise ValueError('actual camera angular truth outside valid range')
    width, height = actual.get('framebuffer_width'), actual.get('framebuffer_height')
    if type(width) is not int or type(height) is not int or width < 1 or height < 1 or [height, width] != shape[:2]:
        raise ValueError('actual camera framebuffer truth mismatch')
    if 'roll_degrees' in actual:
        roll = number(actual['roll_degrees'])
        roll_source = 'recorded_actual_roll'
    elif actual.get('shader_pack_applied') is False and actual.get('active_resource_packs') == ['vanilla']:
        roll = 0.
        roll_source = 'vanilla_shader_free_engine_zero_roll_assumption'
    else:
        raise ValueError('actual camera roll truth unavailable')
    return CameraModel(position, yaw, pitch, fov_degrees=fov, eye_height=0., roll_degrees=roll), roll_source


def evaluate_truth(report, results, solution, capture_path, ground_truth_path):
    """Evaluator-only phase; production calls have already finished."""
    from mc_photogrammetry.dataset.blocks import parse_ground_truth, volume_from_ground_truth
    from mc_photogrammetry.evaluation_pair_correspondence import raycast_surface, classify_returned_pair
    if 'session_sha256' in report and report['session_sha256'] != getattr(solution, 'session_sha256', None):
        raise ValueError('image camera report session join mismatch')
    capture, capture_hash = _truth_snapshot(capture_path, 2 * 1024 * 1024)
    if type(capture) is not dict or type(capture.get('frames')) is not list:
        raise ValueError('actual capture truth schema')
    if len(capture['frames']) != len(report['frames']):
        raise ValueError('truth frame count mismatch')
    filenames = [frame.get('filename') for frame in capture['frames'] if type(frame) is dict]
    if len(filenames) != len(capture['frames']) or any(type(name) is not str or not name for name in filenames) or len(set(filenames)) != len(filenames):
        raise ValueError('truth capture identities incomplete or duplicated')
    if [row['frame_id'] for row in report['frames']] != [f'image-{index:04d}' for index in range(len(report['frames']))]:
        raise ValueError('truth capture ordered frame join')
    if 'input_seals' in report:
        screenshots = capture.get('screenshots')
        if type(screenshots) is not list or len(screenshots) != len(filenames):
            raise ValueError('truth screenshot inventory incomplete')
        recorded = {row['filename']: row['sha256'].upper() for row in screenshots}
        if len(recorded) != len(screenshots) or set(recorded) != set(filenames):
            raise ValueError('truth screenshot identities duplicated or foreign')
        for seal, frame, image in zip(report['input_seals'], capture['frames'], report['frames'], strict=True):
            actual = frame.get('actual')
            if type(actual) is not dict:
                raise ValueError('actual camera truth missing')
            if (Path(seal['path']).name != frame['filename']
                or recorded.get(frame['filename']) != seal['sha256']
                or actual['screenshot_sha256'].upper() != seal['sha256']
                or actual['screenshot_bytes'] != seal['bytes']
                or [actual['framebuffer_height'], actual['framebuffer_width']] != image['shape'][:2]):
                raise ValueError('truth pixel source join mismatch')
    truth, truth_camera_sources, snapshots = {}, [], {}
    for index, (frame, image) in enumerate(zip(capture['frames'], report['frames'], strict=True)):
        camera, roll_source = _actual_truth_camera(frame, image['shape'])
        frame_id = f'image-{index:04d}'
        truth[frame_id] = camera
        snapshot = _camera_snapshot(frame_id, solution.cameras.get(frame_id),
                                    getattr(solution, 'session_sha256', None))
        if 'image_camera' in image and image['image_camera'] != snapshot:
            raise ValueError('image camera snapshot source/session join mismatch')
        image['image_camera'] = snapshot
        snapshots[frame_id] = snapshot
        image['calibration_truth_error'] = _frame_calibration_error(snapshot, camera, roll_source, image['shape'])
        truth_camera_sources.append({'frame_id': f'image-{index:04d}', 'pose': 'recorded_actual_eye_yaw_pitch',
            'roll': roll_source, 'intrinsics': 'actual_vertical_FOV_and_framebuffer_centered_square_pixels_uncropped'})
    for left, right in combinations(truth.values(), 2):
        if np.linalg.norm(np.asarray(right.position) - np.asarray(left.position)) <= 1e-12:
            raise ValueError('degenerate actual truth baseline')
    ground, ground_hash = _truth_snapshot(ground_truth_path, 64 * 1024 * 1024)
    occupancy = {coordinate: observation.state for coordinate, observation in volume_from_ground_truth(parse_ground_truth(ground)).items()}
    for row in report['pairs']:
        key = row['left_id'], row['right_id']
        row['camera_family_truth_error'] = _pair_camera_family_error(
            snapshots[key[0]], snapshots[key[1]], truth[key[0]], truth[key[1]])
        result = results[row['arm']].get(key)
        if result is None:
            continue
        left, right = truth[key[0]], truth[key[1]]
        left_r, right_r = camera_rotation_matrix(left), camera_rotation_matrix(right)
        # Aggregate unique pixel pairs only; no feature-index inflation.
        verdicts = Counter()
        identities = set()
        for match in result.matches:
            identity = tuple(float(value) for value in match)
            if identity in identities:
                continue
            identities.add(identity)
            width_l, height_l = solution.cameras[key[0]].intrinsics.image_size
            width_r, height_r = solution.cameras[key[1]].intrinsics.image_size
            hit_l = raycast_surface(key[0], left, width_l, height_l, occupancy, *match[:2]).hit
            hit_r = raycast_surface(key[1], right, width_r, height_r, occupancy, *match[2:]).hit
            verdicts[classify_returned_pair(hit_l, hit_r)] += 1
        row['truth_geometry'] = {'unique_selected': len(identities), 'surface_verdicts': dict(verdicts),
            'same_surface_precision': verdicts['positive'] / len(identities) if identities else None,
            'tolerance_blocks': 1 / 16, 'scope': 'selected_pair_surfaces_not_visible_block_F1'}
        row['pose_error'] = None
        row['correct_accepted'] = False
        if result.solved:
            expected = right_r @ left_r.T
            rotation_error = float(np.degrees(np.arccos(np.clip((np.trace(expected.T @ result.pose.relative_rotation) - 1) / 2, -1., 1.))))
            delta = np.asarray(right.position) - np.asarray(left.position)
            # Compare directions in left-camera coordinates; no yaw gauge fitting.
            baseline_error = _angle(_rotation(solution.cameras[key[0]]) @ result.pose.baseline_direction, left_r @ delta)
            row['pose_error'] = {'rotation_degrees': rotation_error, 'baseline_degrees': baseline_error}
            row['correct_accepted'] = rotation_error <= 5. and baseline_error <= 10.
    summaries = {}
    ids = [row['frame_id'] for row in report['frames']]
    for arm in ('baseline', 'spatial'):
        rows = [row for row in report['pairs'] if row['arm'] == arm]
        accepted = [row for row in rows if row['status'] == 'solved']
        correct = [row for row in accepted if row['correct_accepted']]
        summaries[arm] = {'accepted_pairs': len(accepted), 'correct_accepted_pairs': len(correct),
            'false_accepted_pairs': len(accepted) - len(correct), 'refusals': dict(Counter(row['reason'] for row in rows if row['status'] != 'solved')),
            'correct_edge_components': components(ids, [(row['left_id'], row['right_id']) for row in correct]),
            'connectivity_scope': 'topological_correct_pose_edges_not_cycle_or_scale_qualification'}
        summaries[arm]['all_frames_correctly_connected'] = len(summaries[arm]['correct_edge_components']) == 1
    report['summary'] = summaries
    report['evaluation_capture_sha256'] = capture_hash
    report['evaluation_ground_truth_sha256'] = ground_hash
    report['truth_policy'] = {'read_after_all_matching': True, 'production_inputs': 'pixels+GUI_usable_mask+image_derived_cameras',
        'correct_pose_cutoffs_evaluation_only': {'rotation_degrees': 5., 'baseline_degrees': 10.},
        'one_development_scene_not_holdout': True, 'visible_block_geometry_F1': None, 'exact_state_match': None,
        'delivery_metric_reason': 'pair_matcher_experiment_does_not_run_reconstruction_or_export',
        'camera_sources': truth_camera_sources, 'planned_camera_fallback': False,
        'snapshot_policy': 'bounded_single_read_hash_of_exact_bytes_parsed',
        'camera_adequacy': {
            'scope': 'evaluation_only_after_both_detector_arms_finish_never_selects_production_pose',
            'frame_rotation_gauge': 'min_k d_SO3(R_actual,R_estimated@Ry_world(90*k)),k=0..3',
            'pair_rotation_family': 'R_right_estimated@Ry_world(90*k)@R_left_estimated.T,k=0..3',
            'yaw_error': 'estimate_minus_actual_mod90_in[-45,45);positive45_tie_maps_to_negative45',
            'pitch_error': 'estimate_minus_actual_degrees',
            'roll_error': 'estimate_minus_actual_mod360_in[-180,180);truth_source_explicit',
            'focal_error_relative': '(estimated_focal-actual_focal)/actual_focal,separate_fx_fy_fractions',
            'best_class_tie': 'first_minimum_class_index',
            'common_world_gauge': 'common_world_Y_rotation;arbitrary_tilt_changes_declared_vertical_axis',
            'rotation_cutoff_degrees': 5., 'missing_image_camera': 'unavailable_not_zero_error'}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-manifest', required=True)
    parser.add_argument('--evaluation-capture', required=True)
    parser.add_argument('--evaluation-ground-truth', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise ValueError('refuse overwrite')
    manifest_raw = Path(args.input_manifest).read_bytes()
    manifest = json.loads(manifest_raw)
    inputs = manifest['inputs']
    if not 2 <= len(inputs) <= 6:
        raise ValueError('input manifest frame budget')
    if [seal.get('ordinal') for seal in inputs] != list(range(len(inputs))) or len({seal['path'] for seal in inputs}) != len(inputs):
        raise ValueError('input manifest ordering/identity')
    from PIL import Image
    from io import BytesIO
    frames, seals = {}, []
    start = time.perf_counter()
    for index, seal in enumerate(inputs):
        raw = Path(seal['path']).read_bytes()
        digest = hashlib.sha256(raw).hexdigest().upper()
        if len(raw) != seal['bytes'] or digest != seal['sha256'].upper():
            raise ValueError('input seal mismatch')
        with Image.open(BytesIO(raw)) as image:
            if image.width * image.height > 6_000_000:
                raise ValueError('native pixel budget')
            frames[f'image-{index:04d}'] = np.asarray(image.convert('RGB')).copy()
        seals.append({'path': seal['path'], 'bytes': len(raw), 'sha256': digest})
    report, results, solution = run_pixel_ab(frames)
    # Deliberately deferred: neither exact pose nor occupancy can influence A/B.
    report['input_seals'] = seals
    evaluate_truth(report, results, solution, args.evaluation_capture, args.evaluation_ground_truth)
    report.update(schema='mc-spatial-sift-matcher-ab-v1', diagnostic_only=True, default_promoted=False,
                  input_seals=seals, input_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
                  wall_seconds=time.perf_counter() - start)
    payload = json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(payload) > 1_048_576:
        raise ValueError('report budget')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(payload)
    print(json.dumps({'summary': report['summary'], 'wall_seconds': report['wall_seconds']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
